// agentic-spatial/src/index.js — Core library for spatial reconstruction from photos
// Uses agentic-core's agenticAsk for VLM calls with vision

const IS_BROWSER = typeof window !== 'undefined'
const AGENTIC_CORE_CDN = 'https://momomo-agent.github.io/agentic-core/agentic-agent.js'
const AGENTIC_CORE_LOCAL = '../../agentic-core/docs/agentic-agent.js'

const { agenticAsk } = await import(IS_BROWSER ? AGENTIC_CORE_CDN : AGENTIC_CORE_LOCAL)

const DEFAULT_MODEL = 'claude-sonnet-4-20250514'

// ── Output Schema ──

const SCENE_SCHEMA = {
  type: 'object',
  required: ['room', 'objects', 'people', 'relations', 'cameras'],
  properties: {
    room: {
      type: 'object',
      required: ['shape', 'estimatedWidth', 'estimatedDepth', 'walls'],
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
              side: { type: 'string', description: 'Wall name: north, south, east, west, etc.' },
              features: { type: 'string', description: 'Description of what is on this wall' }
            }
          }
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
          name: { type: 'string', description: 'Descriptive name like "conference table", "whiteboard", "laptop"' },
          type: { type: 'string', enum: ['furniture', 'electronics', 'decoration', 'appliance'] },
          x: { type: 'number', description: '0-1 normalized, left to right' },
          y: { type: 'number', description: '0-1 normalized, floor to ceiling' },
          z: { type: 'number', description: '0-1 normalized, front to back' },
          width: { type: 'number' },
          depth: { type: 'number' },
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
          x: { type: 'number' },
          y: { type: 'number' },
          z: { type: 'number' },
          gazeDegrees: { type: 'number', description: '0=north, 90=east, 180=south, 270=west' },
          gazeTarget: { type: 'string' },
          clothing: { type: 'string' },
          pose: { type: 'string', enum: ['sitting', 'standing', 'leaning'] },
          seenIn: { type: 'array', items: { type: 'number' } }
        }
      }
    },
    relations: { type: 'array', items: { type: 'string' } },
    cameras: { type: 'array' }
  }
}

// ── Prompts ──

function buildAnalysisPrompt(imageCount) {
  return `You are a spatial analysis expert. You are given ${imageCount} photo(s) taken from different camera positions in the same room/space.

For EACH image, analyze:

1. **Room structure**: Shape (rectangular, L-shaped, etc.), approximate dimensions in meters, wall features (doors, windows, whiteboards, screens).

2. **Objects**: Every significant object — furniture (tables, chairs, shelves), electronics (monitors, phones, projectors), decorations, appliances. For each:
   - Name and type
   - Approximate position in the room (left/center/right, near/far, floor/table/wall level)
   - Approximate size
   - Color/material
   - Confidence level (how certain you are)

3. **People**: Every person visible. For each:
   - Position in the room
   - Body pose (sitting, standing, leaning)
   - Gaze direction (which direction they face, and what they're looking at)
   - Clothing description
   - Confidence level

4. **Camera perspective**: Where this camera is positioned relative to the room (which corner/wall), approximate field of view.

5. **Cross-image correspondence**: If the same object or person appears in multiple images, note that explicitly. Use consistent naming.

Be extremely thorough. Describe spatial relationships precisely. Use cardinal directions (north/south/east/west) consistently — pick "north" as the wall the first camera faces.

Respond with detailed analysis for each image, then a summary of the unified space.`
}

function buildReconstructionPrompt(analysisText, imageCount) {
  return `You are a spatial reconstruction system. Based on the per-image analysis below, produce a unified 3D scene reconstruction as JSON.

## Per-Image Analysis
${analysisText}

## Instructions

Merge all observations into one coherent spatial model:

1. **Room**: Determine the room shape and dimensions in meters. List each wall and its features.

2. **Objects**: Place every object using normalized coordinates (0-1 range):
   - x: left(0) to right(1)
   - y: floor(0) to ceiling(1)
   - z: near/front(0) to far/back(1)
   - width, depth: normalized footprint
   - Assign a hex color matching the object's actual color
   - List which camera indices (0-based) saw it
   - Confidence 0-1

3. **People**: Place each person with:
   - x, y, z position (normalized 0-1)
   - gazeDegrees: 0=north, 90=east, 180=south, 270=west
   - gazeTarget: id of the object they face (if any)
   - clothing description
   - pose: sitting/standing/etc
   - seenIn: camera indices

4. **Relations**: Natural language descriptions of spatial relationships between objects and people.

5. **Cameras**: For each of the ${imageCount} input images, estimate:
   - index (0-based)
   - estimatedPosition {x, z} in normalized coords
   - fovDegrees

6. **Walls**: Each wall must have a "side" (e.g. "north", "south", "east", "west") and "features" (string description of what's on it).

Types for objects: furniture, electronics, decoration, appliance
People are separate from objects.

Object IDs: obj_1, obj_2, etc.
People IDs: person_1, person_2, etc.
Object names: use descriptive names like "conference table", "whiteboard", "laptop", NOT generic "obj 1".

CRITICAL POSITIONING RULES:
- Place objects and people so they make spatial sense — objects on tables should have appropriate y values, people sitting in chairs should be near chairs.
- SPREAD PEOPLE OUT: If multiple people are sitting around a table, distribute them along the table edges. Each person must have DISTINCT x,z coordinates with at least 0.05 separation. Do NOT cluster everyone at the same x,z position.
- Example: 6 people around a rectangular table should be placed at 3 positions along each long side, spaced evenly.`
}

// ── Main Export ──

export async function reconstructSpace({ images, apiKey, model, baseUrl, proxyUrl, onProgress }) {
  if (!images?.length) throw new Error('At least one image is required')
  if (!apiKey) throw new Error('API key is required')

  const startTime = Date.now()
  const mdl = model || DEFAULT_MODEL
  const base = baseUrl || 'https://api.anthropic.com'
  const proxy = proxyUrl || undefined
  const provider = base.includes('anthropic.com') ? 'anthropic' : 'openai'
  const progress = onProgress || (() => {})

  progress('start', { imageCount: images.length, model: mdl })

  // ── Step 1: Per-image analysis ──
  progress('step1', { message: 'Analyzing images with VLM...' })

  const visionImages = images.map(img => ({
    data: img.data,
    media_type: img.media_type || 'image/jpeg'
  }))

  const analysisPrompt = buildAnalysisPrompt(images.length)
  const imageNames = images.map((img, i) => img.name || `camera-${i}`).join(', ')

  const step1Result = await agenticAsk(
    `${analysisPrompt}\n\nThe ${images.length} images are named: ${imageNames}. Analyze them all.`,
    {
      provider,
      apiKey,
      model: mdl,
      baseUrl: base,
      proxyUrl: proxy,
      tools: [],
      stream: false,
      images: visionImages
    },
    (type, data) => {
      if (type === 'status') progress('step1_status', data)
    }
  )

  const analysisText = step1Result.answer
  progress('step1_done', { length: analysisText.length })

  // ── Step 2: Unified reconstruction (structured output) ──
  progress('step2', { message: 'Reconstructing unified scene...' })

  const reconstructionPrompt = buildReconstructionPrompt(analysisText, images.length)

  const step2Result = await agenticAsk(
    reconstructionPrompt,
    {
      provider,
      apiKey,
      model: mdl,
      baseUrl: base,
      proxyUrl: proxy,
      tools: [],
      stream: false,
      schema: SCENE_SCHEMA
    },
    (type, data) => {
      if (type === 'status') progress('step2_status', data)
    }
  )

  const scene = step2Result.data

  // ── Attach metadata ──
  scene.meta = {
    model: mdl,
    imageCount: images.length,
    elapsedMs: Date.now() - startTime
  }

  progress('done', { scene })
  return scene
}
