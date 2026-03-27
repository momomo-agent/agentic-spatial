// agentic-spatial/src/index.js — Core library for spatial reconstruction from photos
// Uses agentic-core's agenticAsk for VLM calls with vision
// v7 soft-grid prompt + optional ensemble mode (3x parallel + merge)

const IS_BROWSER = typeof window !== 'undefined'
const AGENTIC_CORE_CDN = 'https://momomo-agent.github.io/agentic-core/agentic-agent.js'
const AGENTIC_CORE_LOCAL = '../../agentic-core/docs/agentic-agent.js'

const { agenticAsk } = await import(IS_BROWSER ? AGENTIC_CORE_CDN : AGENTIC_CORE_LOCAL)

const DEFAULT_MODEL = 'claude-sonnet-4-20250514'

// ── Output Schemas ──

// Step 1: Fixed reference frame — room structure, anchors, cameras
const STEP1_SCHEMA = {
  type: 'object',
  required: ['room', 'anchors', 'cameras'],
  properties: {
    room: {
      type: 'object',
      required: ['shape', 'estimatedWidth', 'estimatedDepth', 'walls', 'grid'],
      properties: {
        shape: { type: 'string' },
        estimatedWidth: { type: 'number' },
        estimatedDepth: { type: 'number' },
        walls: {
          type: 'array',
          items: {
            type: 'object',
            required: ['side', 'features'],
            properties: {
              side: { type: 'string' },
              features: { type: 'string' }
            }
          }
        },
        grid: { type: 'object' }
      }
    },
    anchors: { type: 'array' },
    cameras: { type: 'array' }
  }
}

// Step 2: Dynamic elements positioned relative to anchors
const STEP2_SCHEMA = {
  type: 'object',
  required: ['objects', 'people', 'relations'],
  properties: {
    objects: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'name', 'type', 'x', 'y', 'z'],
        properties: {
          id: { type: 'string' },
          name: { type: 'string' },
          type: { type: 'string', enum: ['furniture', 'electronics', 'decoration', 'appliance'] },
          zone: { type: 'string' },
          anchorRef: { type: 'string' },
          x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number' },
          width: { type: 'number' }, depth: { type: 'number' },
          color: { type: 'string' },
          confidence: { type: 'number' },
          seenIn: { type: 'array', items: { type: 'number' } }
        }
      }
    },
    people: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'x', 'y', 'z', 'pose'],
        properties: {
          id: { type: 'string' },
          zone: { type: 'string' },
          anchorRef: { type: 'string' },
          x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number' },
          gazeDegrees: { type: 'number' },
          gazeTarget: { type: 'string' },
          lookingAtCamera: { type: 'number', description: 'Camera index (0-based) the person is looking at, or -1 if not looking at any camera' },
          clothing: { type: 'string' },
          pose: { type: 'string', enum: ['sitting', 'standing', 'leaning'] },
          emotion: { type: 'string', description: 'Detected emotional state: neutral|happy|focused|bored|confused|surprised|anxious|sad|angry|excited|contemplative' },
          emotionConfidence: { type: 'number', description: '0-1 confidence in emotion detection' },
          activity: { type: 'string', description: 'What this person is doing: typing|reading|talking|listening|presenting|writing|watching|walking|eating|phone_use|vr_use|idle' },
          interactingWith: { type: 'array', items: { type: 'string' }, description: 'IDs of other people this person is interacting with' },
          seenIn: { type: 'array', items: { type: 'number' } }
        }
      }
    },
    relations: { type: 'array', items: { type: 'string' } },
    behaviors: {
      type: 'array',
      items: {
        type: 'object',
        required: ['type'],
        properties: {
          type: { type: 'string', description: 'meeting|presenting|solo_work|conversation|idle|moving|aware_of_camera' },
          participants: { type: 'array', items: { type: 'string' }, description: 'person ids involved' },
          focus: { type: 'string', description: 'anchor/object id that is the focus of this activity' },
          description: { type: 'string' }
        }
      }
    },
    attentionMap: {
      type: 'object',
      description: 'Map of object/anchor id → array of person ids looking at it'
    },
    coverage: {
      type: 'object',
      properties: {
        cameras: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              index: { type: 'number' },
              visiblePeople: { type: 'array', items: { type: 'string' } },
              occludedPeople: { type: 'array', items: { type: 'string' } }
            }
          }
        },
        blindSpots: { type: 'array', items: { type: 'string' }, description: 'Areas not visible from any camera' }
      }
    }
  }
}

// Combined schema used by ensembleMerge and SpatialSession update
const SCENE_SCHEMA = {
  type: 'object',
  required: ['room', 'anchors', 'objects', 'people', 'relations', 'cameras'],
  properties: {
    ...STEP1_SCHEMA.properties,
    ...STEP2_SCHEMA.properties
  }
}

// ── Step 1 Prompt: Fixed reference frame (room + anchors + cameras) ──

function buildStep1Prompt(imageCount) {
  return `Analyze these ${imageCount} photos of the same room taken from different angles. Output ONLY the fixed reference frame: room structure, anchor objects, and camera positions.

COORDINATE SYSTEM (normalized 0-1):
- x: left(0) → right(1)
- y: floor(0) → ceiling(1)
- z: front/north(0) → back/south(1)
- "north" = wall the first camera faces (z=0 side)

YOUR TASK: Establish the spatial skeleton that everything else will be positioned against.

━━━ ROOM (absolute reference frame) ━━━
Analyze the room itself first. This is the fixed coordinate system.
- Shape, dimensions, walls with features
- Divide into 3×3 grid zones: NW, N, NE, W, C, E, SW, S, SE
- Output room.grid describing what occupies each zone

━━━ ANCHORS (fixed objects, 2-4) ━━━
Large immovable objects that define the spatial skeleton. Position them relative to the ROOM.
- id: anchor_{type}_{zone}, type: table|board|door|cabinet|screen
- These don't move — their positions are ground truth for everything else
- x, y, z, width, depth: 0-1, confidence: ~1.0, seenIn: [indices]

━━━ CAMERAS ━━━
Each input image comes from a DIFFERENT camera/device. Output one camera entry per input image.
- index: matches the image index (0-based)
- name: the device name (from image label)
- estimatedPosition: {x, z} where the camera is in the room (0-1)
- facingDegrees, fovDegrees

RULES:
- First determine the room shape and walls. Then find 2-4 large furniture pieces as anchors.
- Zone is a LABEL for reasoning, not a hard coordinate constraint. Place items where they actually are.
- IDs use zone labels for determinism: same object in same zone = same ID every time.
- CONSISTENCY PRIORITY: If an object is near a zone boundary, always assign it to the zone where its CENTER falls. Anchor positions should be identical across any analysis of these images.
- Relations = plain strings, walls = {side, features}.
- Output ONLY room, anchors, cameras. Do NOT include objects, people, or behaviors.`
}

// ── Step 2 Prompt: Dynamic elements relative to anchors ──

function buildStep2Prompt(imageCount, step1Result) {
  return `Analyze these ${imageCount} photos of the same room. Position all dynamic elements (objects, people, behaviors) relative to the known room structure and anchors.

COORDINATE SYSTEM (normalized 0-1):
- x: left(0) → right(1)
- y: floor(0) → ceiling(1)
- z: front/north(0) → back/south(1)
- "north" = wall the first camera faces (z=0 side)

━━━ OBJECTS (relative to anchors) ━━━
Smaller items. Position each relative to its nearest ANCHOR.
- id: {type}_{zone}_{number}, anchorRef: nearest anchor id
- Think: "this laptop is on the LEFT side of anchor_table_C, about 30% from the table edge"
- x, y, z, width, depth: 0-1, confidence: 0-1, seenIn: [indices]

━━━ PEOPLE (relative to anchors + objects) ━━━
Count people using a SYSTEMATIC SCAN — go image by image, zone by zone:
1. For each image: scan NW→N→NE→W→C→E→SW→S→SE
2. In each zone: count heads, shoulders, hands, even partial bodies at edges
3. Cross-reference across images: same person seen from different angles = same ID
4. Final tally: list every unique person with their zone

CRITICAL CROSS-IMAGE DEDUPLICATION:
When multiple images show the same room from different angles, the SAME person appears in multiple images.
You MUST deduplicate:
- Match by POSITION: person at similar spatial coordinates across images = same person
- Match by CLOTHING: same shirt/pants color across images = likely same person
- Match by CONTEXT: person sitting at the same desk/chair across images = same person
- When in doubt, MERGE (fewer false positives is better than inflated count)
- The total people count should reflect UNIQUE individuals in the room, NOT total appearances across all images
- Use seenIn: [image indices] to track which images each person appears in

PEOPLE COUNTING CHECKLIST:
□ Did I check all 9 zones in every image?
□ Did I check edges and partially visible people?
□ Did I check behind large objects where someone might be partially occluded?
□ Did I cross-reference across camera angles?
□ Did I DEDUPLICATE — is each person_id a UNIQUE individual, not a duplicate from another angle?
□ Is my total count ≤ the MINIMUM count seen in any single image that covers the whole room?

Each person needs:
- id: person_{zone}_{number}, zone, anchorRef
- x, y, z: 0-1 (position relative to nearest anchor)
- gazeDegrees, gazeTarget, clothing, pose (sitting|standing|leaning), seenIn
- lookingAtCamera: camera index if facing camera, -1 otherwise
- emotion: neutral|happy|focused|bored|confused|surprised|anxious|sad|angry|excited|contemplative
- emotionConfidence: 0-1
- activity: typing|reading|talking|listening|presenting|writing|watching|walking|eating|phone_use|vr_use|idle
- interactingWith: [person ids]

━━━ BEHAVIORS (emergent from people + objects) ━━━
What's happening? Infer from people positions, gaze, activity.
- type: meeting|presenting|solo_work|conversation|idle|moving|aware_of_camera
- participants: [person ids], focus: anchor/object id, description

━━━ ATTENTION + COVERAGE ━━━
- ATTENTION MAP: object/anchor → [person ids looking at it]
- COVERAGE: per camera: visiblePeople, occludedPeople, blindSpots

RELATIONS: Spatial strings referencing zones/anchors.

RULES:
- PEOPLE COUNT ACCURACY IS CRITICAL. Use the systematic scan above. Only count a person if you can clearly see evidence of a human body (head, torso, or limbs). Do NOT count shadows, reflections, bags, coats on chairs, or ambiguous shapes as people.
- DEDUPLICATION IS CRITICAL. Multiple images show the SAME room from different angles. The same person WILL appear in multiple images. Each person_id must be a UNIQUE individual. When uncertain, assume it's the same person (under-count is safer than over-count).
- USE THE ANCHORS as reference points for positioning. Think: "this person is sitting at anchor_table_C, on the left side".
- Zone is a LABEL for reasoning, not a hard coordinate constraint. Place items where they actually are.
- IDs use zone labels for determinism: same object in same zone = same ID every time.
- SPREAD people: ≥0.05 separation.
- Objects ON a table: y ≈ table.y + 0.06.
- Output ONLY objects, people, relations, behaviors, attentionMap, coverage. Do NOT include room, anchors, or cameras.`
}

// ── Post-processing ──

function postProcess(scene) {
  // 1. Clamp all coordinates to [0, 1]
  const clamp = v => Math.max(0, Math.min(1, v || 0))
  
  for (const obj of (scene.objects || [])) {
    obj.x = clamp(obj.x); obj.y = clamp(obj.y); obj.z = clamp(obj.z)
    // Fallback name
    if (!obj.name) obj.name = obj.id || 'unknown object'
  }
  
  for (const anchor of (scene.anchors || [])) {
    anchor.x = clamp(anchor.x); anchor.y = clamp(anchor.y); anchor.z = clamp(anchor.z)
  }

  // 2. People: clamp + jitter to prevent overlap
  const people = scene.people || []
  for (const p of people) {
    p.x = clamp(p.x); p.y = clamp(p.y); p.z = clamp(p.z)
  }

  const MIN_DIST = 0.05
  for (let i = 0; i < people.length; i++) {
    for (let j = i + 1; j < people.length; j++) {
      const dx = people[j].x - people[i].x
      const dz = people[j].z - people[i].z
      const dist = Math.sqrt(dx * dx + dz * dz)
      if (dist < MIN_DIST) {
        const angle = Math.atan2(dz, dx) || (Math.random() * Math.PI * 2)
        const push = (MIN_DIST - dist) / 2 + 0.01
        people[j].x = clamp(people[j].x + Math.cos(angle) * push)
        people[j].z = clamp(people[j].z + Math.sin(angle) * push)
        people[i].x = clamp(people[i].x - Math.cos(angle) * push)
        people[i].z = clamp(people[i].z - Math.sin(angle) * push)
      }
    }
  }

  // 3. Wall fallback
  if (scene.room?.walls) {
    scene.room.walls = scene.room.walls.map(w => {
      if (typeof w === 'string') return { side: w, features: '' }
      if (!w.side) w.side = w.name || w.direction || w.wall || 'unknown'
      if (!w.features) w.features = w.description || w.content || ''
      return w
    })
  }

  // 4. Behaviors: filter out invalid entries
  if (scene.behaviors) {
    scene.behaviors = scene.behaviors.filter(b => b && b.type && b.description)
  }

  // 5. Changes: filter out invalid entries
  if (scene.changes) {
    scene.changes = scene.changes.filter(c => c && c.type && c.id && c.description)
  }

  // 6. People: ensure emotion/activity defaults
  for (const p of (scene.people || [])) {
    if (!p.emotion) p.emotion = 'neutral'
    if (!p.activity) p.activity = 'idle'
    if (!p.interactingWith) p.interactingWith = []
    if (p.lookingAtCamera === undefined) p.lookingAtCamera = -1
  }

  return scene
}

// ── Ensemble: merge multiple runs via coordinate matching ──

function ensembleMerge(runs) {
  const N = runs.length
  const QUORUM = Math.ceil(N / 2)

  function matchItems(itemsPerRun, distThreshold) {
    const clusters = []
    const used = itemsPerRun.map(() => new Set())

    for (let bi = 0; bi < itemsPerRun[0].length; bi++) {
      const base = itemsPerRun[0][bi]
      const cluster = [{ runIdx: 0, item: base }]
      const bType = (base.type || base.pose || '').toLowerCase()

      for (let r = 1; r < N; r++) {
        let bestDist = Infinity, bestIdx = -1
        for (let j = 0; j < itemsPerRun[r].length; j++) {
          if (used[r].has(j)) continue
          const c = itemsPerRun[r][j]
          const cType = (c.type || c.pose || '').toLowerCase()
          if (base.type && c.type && bType !== cType) continue
          const dx = (c.x || 0) - (base.x || 0)
          const dz = (c.z || 0) - (base.z || 0)
          const dist = Math.sqrt(dx * dx + dz * dz)
          if (dist < bestDist) { bestDist = dist; bestIdx = j }
        }
        if (bestDist < distThreshold && bestIdx >= 0) {
          cluster.push({ runIdx: r, item: itemsPerRun[r][bestIdx] })
          used[r].add(bestIdx)
        }
      }
      clusters.push(cluster)
    }

    // Pick up unmatched items from other runs
    for (let r = 1; r < N; r++) {
      for (let j = 0; j < itemsPerRun[r].length; j++) {
        if (used[r].has(j)) continue
        const item = itemsPerRun[r][j]
        const iType = (item.type || item.pose || '').toLowerCase()
        let bestCi = -1, bestDist = Infinity
        for (let ci = 0; ci < clusters.length; ci++) {
          if (clusters[ci].some(e => e.runIdx === r)) continue
          const rep = clusters[ci][0].item
          const rType = (rep.type || rep.pose || '').toLowerCase()
          if (item.type && rep.type && iType !== rType) continue
          const dx = (item.x || 0) - (rep.x || 0)
          const dz = (item.z || 0) - (rep.z || 0)
          const dist = Math.sqrt(dx * dx + dz * dz)
          if (dist < bestDist) { bestDist = dist; bestCi = ci }
        }
        if (bestDist < distThreshold && bestCi >= 0) {
          clusters[bestCi].push({ runIdx: r, item })
        } else {
          clusters.push([{ runIdx: r, item }])
        }
      }
    }

    return clusters
  }

  function median(arr) {
    const sorted = [...arr].sort((a, b) => a - b)
    const mid = Math.floor(sorted.length / 2)
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
  }

  function mostCommon(arr) {
    const freq = {}
    arr.forEach(v => { if (v) freq[v] = (freq[v] || 0) + 1 })
    const entries = Object.entries(freq)
    return entries.length ? entries.sort((a, b) => b[1] - a[1])[0][0] : undefined
  }

  function mergeCluster(cluster, kind) {
    const items = cluster.map(c => c.item)
    const out = {}
    out.x = +median(items.map(i => i.x || 0)).toFixed(3)
    out.y = +median(items.map(i => i.y || 0)).toFixed(3)
    out.z = +median(items.map(i => i.z || 0)).toFixed(3)

    const run0 = cluster.find(c => c.runIdx === 0)
    out.id = (run0 || cluster[0]).item.id
    out.name = items.sort((a, b) => (b.name || '').length - (a.name || '').length)[0].name || out.id

    if (kind === 'object') {
      out.width = +median(items.map(i => i.width || 0.05)).toFixed(3)
      out.depth = +median(items.map(i => i.depth || 0.05)).toFixed(3)
      out.type = mostCommon(items.map(i => i.type))
      out.color = mostCommon(items.map(i => i.color))
      out.confidence = +median(items.map(i => i.confidence || 0.8)).toFixed(2)
      out.anchorRef = mostCommon(items.map(i => i.anchorRef))
      out.zone = mostCommon(items.map(i => i.zone))
    }

    if (kind === 'person') {
      out.pose = mostCommon(items.map(i => i.pose))
      out.clothing = items[0].clothing
      out.gazeDegrees = +median(items.map(i => i.gazeDegrees || 0)).toFixed(0)
      out.gazeTarget = mostCommon(items.map(i => i.gazeTarget))
      out.lookingAtCamera = +median(items.map(i => i.lookingAtCamera ?? -1)).toFixed(0)
      out.emotion = mostCommon(items.map(i => i.emotion))
      out.emotionConfidence = +median(items.map(i => i.emotionConfidence || 0.5)).toFixed(2)
      out.activity = mostCommon(items.map(i => i.activity))
      out.interactingWith = [...new Set(items.flatMap(i => i.interactingWith || []))]
      out.zone = mostCommon(items.map(i => i.zone))
      out.anchorRef = mostCommon(items.map(i => i.anchorRef))
    }

    const seenSet = new Set()
    items.forEach(i => (i.seenIn || []).forEach(s => seenSet.add(s)))
    out.seenIn = [...seenSet].sort()

    return out
  }

  // Merge objects
  const objClusters = matchItems(runs.map(r => r.objects || []), 0.15)
  const objects = objClusters.filter(c => c.length >= QUORUM).map(c => mergeCluster(c, 'object'))

  // Merge people
  const pplClusters = matchItems(runs.map(r => r.people || []), 0.20)
  const people = pplClusters.filter(c => c.length >= QUORUM).map(c => mergeCluster(c, 'person'))

  // Merge anchors
  const anchorArrays = runs.map(r => r.anchors || [])
  let anchors = []
  if (anchorArrays.some(a => a.length > 0)) {
    const anchorClusters = matchItems(anchorArrays, 0.15)
    anchors = anchorClusters.filter(c => c.length >= QUORUM).map(c => mergeCluster(c, 'object'))
  }

  // Room: median dimensions
  const room = {
    shape: runs[0].room?.shape || 'rectangular',
    estimatedWidth: +median(runs.map(r => r.room?.estimatedWidth || 7)).toFixed(1),
    estimatedDepth: +median(runs.map(r => r.room?.estimatedDepth || 5)).toFixed(1),
    walls: runs[0].room?.walls || [],
    grid: runs[0].room?.grid
  }

  // Cameras: median
  const maxCams = Math.max(...runs.map(r => (r.cameras || []).length))
  const cameras = []
  for (let i = 0; i < maxCams; i++) {
    const cams = runs.map(r => (r.cameras || [])[i]).filter(Boolean)
    if (cams.length >= QUORUM) {
      cameras.push({
        index: i,
        estimatedPosition: {
          x: +median(cams.map(c => c.estimatedPosition?.x || 0)).toFixed(2),
          z: +median(cams.map(c => c.estimatedPosition?.z || 0)).toFixed(2)
        },
        fovDegrees: +median(cams.map(c => c.fovDegrees || 60)).toFixed(0)
      })
    }
  }

  // Relations: union
  const relSet = new Set()
  runs.forEach(r => (r.relations || []).forEach(rel => {
    if (typeof rel === 'string') relSet.add(rel)
  }))

  return { room, anchors, objects, people, relations: [...relSet], cameras }
}

// ── Two-step LLM call ──

async function runOnce({ images, apiKey, model, baseUrl, proxyUrl, provider: providerOverride, onProgress }) {
  const mdl = model || DEFAULT_MODEL
  const base = baseUrl || 'https://api.anthropic.com'
  const proxy = proxyUrl || undefined
  const provider = providerOverride || (base.includes('anthropic.com') ? 'anthropic' : 'openai')
  const progress = onProgress || (() => {})

  const visionImages = images.map(img => ({
    data: img.data,
    media_type: img.media_type || 'image/jpeg',
    detail: 'high'
  }))

  // ── Step 1: Fixed reference frame (room + anchors + cameras) ──
  progress('step1_start', { message: 'Establishing room structure and anchors...' })

  const step1Prompt = buildStep1Prompt(images.length)
  const step1SchemaStr = JSON.stringify(STEP1_SCHEMA, null, 2)

  const step1Result = await agenticAsk(
    step1Prompt,
    {
      provider,
      apiKey,
      model: mdl,
      baseUrl: base,
      proxyUrl: proxy,
      tools: [],
      stream: false,
      schema: STEP1_SCHEMA,
      systemPrompt: `You must respond with valid JSON matching this schema:\n${step1SchemaStr}\n\nOutput ONLY JSON, no markdown, no code fences, no explanation.`,
      images: visionImages
    },
    (type, data) => {
      if (type === 'status') progress('step1_status', data)
    }
  )

  const step1 = step1Result.data
  progress('step1_done', { room: step1.room, anchorCount: (step1.anchors || []).length, cameraCount: (step1.cameras || []).length })

  // ── Step 2: Dynamic elements positioned relative to anchors ──
  progress('step2_start', { message: 'Positioning objects and people relative to anchors...' })

  const step2Prompt = buildStep2Prompt(images.length, step1)
  const step2SchemaStr = JSON.stringify(STEP2_SCHEMA, null, 2)

  const step2SystemPrompt = `你已经知道这个房间的结构：\n${JSON.stringify(step1, null, 2)}\n\n基于这些锚点，定位以下动态元素。\n\nYou must respond with valid JSON matching this schema:\n${step2SchemaStr}\n\nOutput ONLY JSON, no markdown, no code fences, no explanation.`

  const step2Result = await agenticAsk(
    step2Prompt,
    {
      provider,
      apiKey,
      model: mdl,
      baseUrl: base,
      proxyUrl: proxy,
      tools: [],
      stream: false,
      schema: STEP2_SCHEMA,
      systemPrompt: step2SystemPrompt,
      images: visionImages
    },
    (type, data) => {
      if (type === 'status') progress('step2_status', data)
    }
  )

  const step2 = step2Result.data
  progress('step2_done', { objectCount: (step2.objects || []).length, peopleCount: (step2.people || []).length })

  // ── Merge step 1 + step 2 into complete scene ──
  const scene = {
    room: step1.room,
    anchors: step1.anchors || [],
    cameras: step1.cameras || [],
    objects: step2.objects || [],
    people: step2.people || [],
    relations: step2.relations || [],
    behaviors: step2.behaviors || [],
    attentionMap: step2.attentionMap || {},
    coverage: step2.coverage || {}
  }

  return postProcess(scene)
}

// ── Main Export ──

export async function reconstructSpace({ images, apiKey, model, baseUrl, proxyUrl, provider, ensemble, onProgress }) {
  if (!images?.length) throw new Error('At least one image is required')
  if (!apiKey) throw new Error('API key is required')

  const startTime = Date.now()
  const progress = onProgress || (() => {})
  const ensembleRuns = ensemble ? (typeof ensemble === 'number' ? ensemble : 3) : 1

  progress('start', { imageCount: images.length, model: model || DEFAULT_MODEL, ensemble: ensembleRuns })

  let scene

  if (ensembleRuns > 1) {
    // Parallel ensemble
    progress('ensemble', { message: `Running ${ensembleRuns} parallel analyses...`, runs: ensembleRuns })

    const promises = Array.from({ length: ensembleRuns }, (_, i) =>
      runOnce({
        images, apiKey, model, baseUrl, proxyUrl, provider,
        onProgress: (step, data) => progress(`run${i}_${step}`, data)
      }).catch(err => {
        progress(`run${i}_error`, { error: err.message })
        return null
      })
    )

    const results = (await Promise.all(promises)).filter(Boolean)
    
    if (results.length === 0) throw new Error('All ensemble runs failed')
    if (results.length === 1) {
      scene = results[0]
      progress('ensemble_partial', { message: 'Only 1 run succeeded, using single result' })
    } else {
      progress('ensemble_merge', { message: `Merging ${results.length} results...` })
      scene = ensembleMerge(results)
      scene = postProcess(scene)
    }
  } else {
    // Single run
    progress('step1', { message: 'Analyzing and reconstructing scene...' })
    scene = await runOnce({ images, apiKey, model, baseUrl, proxyUrl, provider, onProgress: progress })
  }

  // Attach metadata
  scene.meta = {
    model: model || DEFAULT_MODEL,
    imageCount: images.length,
    ensemble: ensembleRuns > 1,
    ensembleRuns: ensembleRuns,
    elapsedMs: Date.now() - startTime
  }

  progress('done', { scene })
  return scene
}

// ── Continuous Mode: SpatialSession ──

function buildUpdatePrompt(prevScene, imageCount) {
  const prevSummary = JSON.stringify({
    room: prevScene.room,
    anchors: (prevScene.anchors || []).map(a => ({ id: a.id, x: a.x, z: a.z })),
    objects: (prevScene.objects || []).map(o => ({ id: o.id, name: o.name, type: o.type, x: o.x, z: o.z })),
    people: (prevScene.people || []).map(p => ({ id: p.id, x: p.x, z: p.z, pose: p.pose, lookingAtCamera: p.lookingAtCamera })),
    behaviors: prevScene.behaviors || []
  })

  return `You are analyzing ${imageCount} NEW photo(s) of the SAME room you analyzed before.

PREVIOUS STATE (your last analysis):
${prevSummary}

TASK: Compare new photos with previous state. Output a FULL updated scene JSON with ALL fields.

CHANGE DETECTION — also include a "changes" array describing what changed:
- "appeared": person/object newly visible (wasn't in previous state)
- "disappeared": person/object no longer visible
- "moved": person/object changed position significantly (>0.1 in normalized coords)
- "pose_changed": person changed pose (sitting→standing, etc.)
- "gaze_changed": person changed gaze direction or lookingAtCamera status
- "emotion_changed": person's emotional state changed
- "activity_changed": person's activity changed  
- "behavior_changed": group behavior type changed

Each change entry: { type, id, description, from, to }

RULES:
- Keep IDs STABLE: if person_N_1 was at (0.3, 0.2) and someone is still near there, keep the same ID.
- Only change an ID if the person is clearly different (different clothing, position too far from previous).
- Room structure and anchors should be kept from previous state unless clearly wrong.
- Coordinates: normalized 0-1. x: left→right, y: floor→ceiling, z: front→back.
- Include ALL standard fields: room, anchors, objects, people, relations, cameras, behaviors, attentionMap, coverage.
- Add "changes" array at top level.`
}

export class SpatialSession {
  constructor({ apiKey, model, baseUrl, proxyUrl, provider, ensemble, onProgress }) {
    this.config = { apiKey, model, baseUrl, proxyUrl, provider, ensemble }
    this.onProgress = onProgress || (() => {})
    this.state = null       // current scene
    this.frameCount = 0
    this.history = []       // array of { timestamp, scene, changes }
  }

  // First analysis or full reset
  async analyze(images) {
    this.frameCount++
    const scene = await reconstructSpace({
      ...this.config,
      images,
      onProgress: this.onProgress
    })
    this.state = scene
    this.history.push({
      frame: this.frameCount,
      timestamp: Date.now(),
      scene,
      changes: []
    })
    return scene
  }

  // Incremental update with new photos
  async update(images) {
    if (!this.state) return this.analyze(images)

    this.frameCount++
    const progress = this.onProgress
    const startTime = Date.now()
    const { apiKey, model, baseUrl, proxyUrl, provider: providerCfg } = this.config
    const mdl = model || DEFAULT_MODEL
    const base = baseUrl || 'https://api.anthropic.com'
    const proxy = proxyUrl || undefined
    const provider = providerCfg || (base.includes('anthropic.com') ? 'anthropic' : 'openai')

    progress('update_start', { frame: this.frameCount, imageCount: images.length })

    const visionImages = images.map(img => ({
      data: img.data,
      media_type: img.media_type || 'image/jpeg',
      detail: 'high'
    }))

    const updatePrompt = buildUpdatePrompt(this.state, images.length)
    const schemaWithChanges = {
      ...SCENE_SCHEMA,
      properties: {
        ...SCENE_SCHEMA.properties,
        changes: {
          type: 'array',
          items: {
            type: 'object',
            required: ['type', 'id', 'description'],
            properties: {
              type: { type: 'string', enum: ['appeared', 'disappeared', 'moved', 'pose_changed', 'gaze_changed', 'behavior_changed'] },
              id: { type: 'string' },
              description: { type: 'string' },
              from: { type: 'string' },
              to: { type: 'string' }
            }
          }
        }
      }
    }

    const result = await agenticAsk(
      updatePrompt,
      {
        provider, apiKey, model: mdl, baseUrl: base, proxyUrl: proxy,
        tools: [], stream: false,
        schema: schemaWithChanges,
        systemPrompt: `You must respond with valid JSON. Output ONLY JSON, no markdown, no code fences.`,
        images: visionImages
      },
      (type, data) => {
        if (type === 'status') progress('update_status', data)
      }
    )

    const newScene = postProcess(result.data)
    const changes = newScene.changes || []

    // Stabilize IDs: match new people to previous by proximity
    if (this.state.people?.length && newScene.people?.length) {
      const prevPeople = this.state.people
      for (const np of newScene.people) {
        let bestDist = Infinity, bestPrev = null
        for (const pp of prevPeople) {
          const dx = (np.x || 0) - (pp.x || 0)
          const dz = (np.z || 0) - (pp.z || 0)
          const dist = Math.sqrt(dx * dx + dz * dz)
          if (dist < bestDist) { bestDist = dist; bestPrev = pp }
        }
        // If close enough and same pose type, keep the old ID
        if (bestDist < 0.15 && bestPrev) {
          np._prevId = bestPrev.id
          if (np.id !== bestPrev.id) np.id = bestPrev.id
        }
      }
    }

    newScene.meta = {
      model: mdl,
      imageCount: images.length,
      frame: this.frameCount,
      elapsedMs: Date.now() - startTime,
      changesDetected: changes.length
    }

    this.state = newScene
    this.history.push({
      frame: this.frameCount,
      timestamp: Date.now(),
      scene: newScene,
      changes
    })

    progress('update_done', { scene: newScene, changes })
    return newScene
  }

  // Get change summary between any two frames
  getChanges(fromFrame, toFrame) {
    const allChanges = []
    for (const entry of this.history) {
      if (entry.frame > fromFrame && entry.frame <= toFrame) {
        allChanges.push(...entry.changes.map(c => ({ ...c, frame: entry.frame })))
      }
    }
    return allChanges
  }

  // Get current state
  getState() { return this.state }

  // Get full history
  getHistory() { return this.history }

  // Reset session
  reset() {
    this.state = null
    this.frameCount = 0
    this.history = []
  }
}
