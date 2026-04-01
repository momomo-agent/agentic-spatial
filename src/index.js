// agentic-spatial/src/index.js — Core library for spatial reconstruction from photos
// Uses agentic-core's agenticAsk for VLM calls with vision
// v7 soft-grid prompt + optional ensemble mode (3x parallel + merge)

const IS_BROWSER = typeof window !== 'undefined'
const AGENTIC_CORE_CDN = 'https://cdn.jsdelivr.net/gh/momomo-agent/agentic-core@main/agentic-core.js'
const AGENTIC_CORE_LOCAL = '../../agentic-core/agentic-core.js'

let agenticAsk
if (IS_BROWSER) {
  // In browser, agentic-core is UMD and exports to window
  await import(AGENTIC_CORE_CDN)
  agenticAsk = window.agenticAsk
  if (!agenticAsk) throw new Error('agenticAsk not found in window after loading agentic-core')
} else {
  const _mod = await import(AGENTIC_CORE_LOCAL)
  agenticAsk = (_mod.default || _mod).agenticAsk
}

const DEFAULT_MODEL = 'claude-sonnet-4-20250514'

// ── Output Schema ──

const SCENE_SCHEMA = {
  type: 'object',
  required: ['room', 'anchors', 'objects', 'people', 'cameras'],
  properties: {
    room: {
      type: 'object',
      required: ['shape', 'estimatedWidth', 'estimatedDepth'],
      properties: {
        shape: { type: 'string' },
        estimatedWidth: { type: 'number' },
        estimatedDepth: { type: 'number' }
      }
    },
    anchors: {
      type: 'array',
      items: {
        type: 'object',
        required: ['id', 'name', 'x', 'y', 'z'],
        properties: {
          id: { type: 'string' },
          name: { type: 'string' },
          type: { type: 'string' },
          x: { type: 'number' }, y: { type: 'number' }, z: { type: 'number' },
          width: { type: 'number' }, depth: { type: 'number' },
          facingDegrees: { type: 'number', description: '0-360, 0=north(z=0), 90=east(x=1), 180=south(z=1), 270=west(x=0)' }
        }
      }
    },
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
          facingDegrees: { type: 'number', description: '0-360, 0=north(z=0), 90=east(x=1), 180=south(z=1), 270=west(x=0)' },
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
          activity: { type: 'string', description: 'What this person is doing: typing|reading|talking|listening|presenting|writing|watching|walking|eating|phone_use|vr_use|idle' },
          interactingWith: { type: 'array', items: { type: 'string' }, description: 'IDs of other people this person is interacting with' },
          seenIn: { type: 'array', items: { type: 'number' } }
        }
      }
    },
    cameras: { type: 'array' },
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
            required: ['index', 'position'],
            properties: {
              index: { type: 'number' },
              position: {
                type: 'object',
                required: ['x', 'y', 'z', 'facingDegrees'],
                properties: {
                  x: { type: 'number', description: '0-1, left to right' },
                  y: { type: 'number', description: '0-1, floor to ceiling' },
                  z: { type: 'number', description: '0-1, north (cam0) to south' },
                  facingDegrees: { type: 'number', description: '0-360, 0=north, 90=east, 180=south, 270=west' }
                }
              },
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

// ── v7 Soft Grid + Anchor Prompt ──

function buildPrompt(imageCount, sensorHints, images) {
  let prompt = `${imageCount} photos, same room, different angles. Reconstruct as JSON.

Coords 0-1: x:left→right, y:floor→ceiling, z:north(cam0)→south.

room → anchors (fixed: tables,screens,doors. Doors MUST be near walls x/z<0.15 or>0.85) → cameras (1 per image) → objects (relative to anchors, on table: y≈table.y+0.06) → people → behaviors → relations → attentionMap → coverage.

CRITICAL - PEOPLE DETECTION: This is the MOST IMPORTANT part. Carefully examine EACH image:
1. Scan systematically left-to-right, top-to-bottom
2. Look for ANY visible human features: heads, faces, shoulders, arms, hands, torsos, legs
3. Count EVERY person, even if partially visible or occluded by objects (laptops, monitors, furniture)
4. People sitting at tables are ALWAYS present - look carefully at each seat position
5. Match the same person across different camera angles using clothing color/style and approximate position
6. If sensor data reports N faces, you MUST find at least N people in the images
7. List EVERY unique individual with their position, pose, activity, and what they're looking at

DO NOT output an empty people array unless the room is genuinely empty. Multiple people in a room is the NORMAL case.

IDs: anchor_{type}_{zone}, {type}_{zone}_{n}, person_{zone}_{n}. People ≥0.05 apart.`

  // Add device mapping
  if (images && images.length > 0) {
    prompt += `\n\nIMAGE-DEVICE MAPPING (use deviceId in cameras array):`
    images.forEach((img, idx) => {
      if (img.deviceId) {
        prompt += `\n- Image ${idx}: deviceId="${img.deviceId}" (${img.name || 'unknown'})`
      }
    })
  }

  if (sensorHints) {
    prompt += `\n\nSENSOR DATA (from device face-detection, use as constraints for people positions and gaze):\n${sensorHints}`
  }

  return prompt
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
        position: {
          x: +median(cams.map(c => c.position?.x || c.estimatedPosition?.x || 0)).toFixed(2),
          z: +median(cams.map(c => c.position?.z || c.estimatedPosition?.z || 0)).toFixed(2),
          facingDegrees: +median(cams.map(c => c.position?.facingDegrees || c.facingDegrees || 0)).toFixed(0)
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

  return { room, anchors, objects, people, cameras }
}

// ── Single LLM call ──

async function runOnce({ images, apiKey, model, baseUrl, proxyUrl, provider: providerOverride, onProgress, sensorHints }) {
  const mdl = model || DEFAULT_MODEL
  const base = baseUrl || 'https://api.anthropic.com'
  const proxy = proxyUrl || undefined
  const provider = providerOverride || (base.includes('anthropic.com') ? 'anthropic' : 'openai')
  const progress = onProgress || (() => {})

  // Debug: log image data sizes
  console.log('[agentic-spatial] Processing images:', images.map((img, i) => ({
    index: i,
    hasData: !!img.data,
    dataLength: img.data?.length || 0,
    dataPrefix: img.data?.substring(0, 50)
  })))

  const visionImages = images.map(img => ({
    data: img.data,
    media_type: img.media_type || 'image/jpeg',
    detail: 'high'
  }))

  const prompt = buildPrompt(images.length, sensorHints, images)
  const schemaStr = JSON.stringify(SCENE_SCHEMA, null, 2)

  const result = await agenticAsk(
    prompt,
    {
      provider,
      apiKey,
      model: mdl,
      baseUrl: base,
      proxyUrl: proxy,
      tools: [],
      stream: false,
      schema: SCENE_SCHEMA,
      systemPrompt: `You must respond with valid JSON matching this schema:\n${schemaStr}\n\nOutput ONLY JSON, no markdown, no code fences, no explanation.`,
      images: visionImages
    },
    (type, data) => {
      if (type === 'status') progress('llm_status', data)
    }
  )

  return postProcess(result.data)
}

// ── Main Export ──

export async function reconstructSpace({ images, apiKey, model, baseUrl, proxyUrl, provider, ensemble, onProgress, sensorHints }) {
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
        images, apiKey, model, baseUrl, proxyUrl, provider, sensorHints,
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
    scene = await runOnce({ images, apiKey, model, baseUrl, proxyUrl, provider, sensorHints, onProgress: progress })
  }

  // Attach metadata
  scene.meta = {
    model: model || DEFAULT_MODEL,
    imageCount: images.length,
    ensemble: ensembleRuns > 1,
    ensembleRuns: ensembleRuns,
    elapsedMs: Date.now() - startTime
  }

  // Attach deviceId to cameras based on image index
  if (scene.cameras && images) {
    console.log('[agentic-spatial] Attaching deviceId to cameras. Images:', images.map(img => ({ index: images.indexOf(img), deviceId: img.deviceId, name: img.name })))
    scene.cameras = scene.cameras.map(cam => {
      const deviceId = images[cam.index]?.deviceId
      console.log(`[agentic-spatial] Camera index=${cam.index}, deviceId=${deviceId}`)
      return {
        ...cam,
        deviceId
      }
    })
  }

  progress('done', { scene })
  return scene
}

// ── Continuous Mode: SpatialSession ──

function buildUpdatePrompt(prevScene, imageCount, sensorHints) {
  const prevSummary = JSON.stringify({
    room: prevScene.room,
    anchors: (prevScene.anchors || []).map(a => ({ id: a.id, x: a.x, z: a.z })),
    objects: (prevScene.objects || []).map(o => ({ id: o.id, name: o.name, type: o.type, x: o.x, z: o.z })),
    people: (prevScene.people || []).map(p => ({ id: p.id, x: p.x, z: p.z, pose: p.pose, lookingAtCamera: p.lookingAtCamera })),
    behaviors: prevScene.behaviors || []
  })

  const base = `You are analyzing ${imageCount} NEW photo(s) of the SAME room you analyzed before.

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

CRITICAL - PEOPLE DETECTION: Carefully examine ALL new photos for people:
- If sensor data reports N faces, you MUST find at least N people
- Look for heads, shoulders, arms, hands - even if partially occluded
- People sitting at desks/tables are common - check each seat position
- DO NOT report 0 people unless the room is genuinely empty

RULES:
- Keep IDs STABLE: if person_N_1 was at (0.3, 0.2) and someone is still near there, keep the same ID.
- Only change an ID if the person is clearly different (different clothing, position too far from previous).
- Room structure and anchors should be kept from previous state unless clearly wrong.
- Coordinates: normalized 0-1. x: left→right, y: floor→ceiling, z: front→back.
- Include ALL standard fields: room, anchors, objects, people, relations, cameras, behaviors, attentionMap, coverage.
- Add "changes" array at top level.`

  if (sensorHints) {
    return base + `\n\nSENSOR DATA (from device face-detection, use as constraints for people positions and gaze):\n${sensorHints}`
  }
  return base
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
  async analyze(images, { sensorHints } = {}) {
    this.frameCount++
    const scene = await reconstructSpace({
      ...this.config,
      images,
      sensorHints,
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
  async update(images, { sensorHints } = {}) {
    if (!this.state) return this.analyze(images, { sensorHints })

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

    const updatePrompt = buildUpdatePrompt(this.state, images.length, sensorHints)
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
