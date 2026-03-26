# agentic-spatial

从多张照片重建空间场景。基于 VLM（Vision Language Model）的空间理解。

Part of the [agentic](https://momomo-agent.github.io/agentic/) family.

## Architecture

```
agentic-spatial/
├── src/
│   └── index.js          # 核心库（ESM），用 agentic-core 调 VLM
├── demo/
│   └── index.html        # 单页 demo，Three.js 3D 可视化
├── test-images/          # 测试图片（symlink）
└── package.json
```

## API

```js
import { reconstructSpace } from './src/index.js'

const scene = await reconstructSpace({
  images: [
    { data: base64String, name: 'camera-1' },
    { data: base64String, name: 'camera-2' },
  ],
  apiKey: 'sk-ant-...',
  model: 'claude-sonnet-4-20250514',  // default
  onProgress: (step, detail) => console.log(step, detail),
})
```

## Output Schema

```js
{
  room: {
    shape: "rectangular",
    estimatedWidth: 8,    // meters
    estimatedDepth: 6,
    walls: [
      { side: "north", features: ["whiteboard", "door"] },
      { side: "east", features: ["windows"] }
    ]
  },
  objects: [
    {
      id: "obj_1",
      name: "conference_table",
      type: "furniture",         // furniture|electronics|decoration|appliance|person
      x: 0.5, y: 0.0, z: 0.4,  // normalized 0-1 (x=left-right, y=height, z=depth)
      width: 0.3, depth: 0.15,  // estimated footprint
      color: "#8B4513",
      seenIn: [0, 1, 2],        // which camera indices
      confidence: 0.9
    }
  ],
  people: [
    {
      id: "person_1",
      x: 0.3, y: 0.0, z: 0.5,
      gazeDegrees: 90,           // 0=north, 90=east, 180=south, 270=west
      gazeTarget: "monitor_1",
      clothing: "blue shirt, dark pants",
      pose: "sitting",
      seenIn: [0, 2]
    }
  ],
  relations: [
    "conference_table is in the center of the room",
    "person_1 is sitting at the table facing monitor_1"
  ],
  cameras: [
    { index: 0, name: "camera-1", estimatedPosition: { x: 0.1, z: 0.9 }, fovDegrees: 90 }
  ],
  meta: {
    model: "claude-sonnet-4-20250514",
    imageCount: 3,
    elapsedMs: 12345
  }
}
```
