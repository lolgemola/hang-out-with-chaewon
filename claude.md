# Face Scanner — Reaction Meme Displayer

## Project overview

A real-time Python app that reads the webcam, detects facial expressions and hand gestures using MediaPipe, and displays a matching fun reaction image/meme on screen. Built for VS Code with a clean modular structure.

---

## Tech stack

| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture, window display, frame compositing |
| `mediapipe` | Face landmark detection (468 points) + hand tracking (21 points) |
| `Pillow` | Loading and resizing reaction images |
| `numpy` | Array ops for frame manipulation |

Install everything with:
```bash
pip install opencv-python mediapipe Pillow numpy
```

---

## Project structure

```
face_scanner/
├── CLAUDE.md              ← you are here
├── main.py                ← entry point: webcam loop, orchestrates all modules
├── classifier.py          ← converts raw landmarks → a string label
├── image_picker.py        ← maps label → random image from the right folder
├── utils.py               ← overlay helper, image resize, cooldown timer
├── reactions/
│   ├── happy/             ← .jpg/.png meme files for each state
│   ├── surprised/
│   ├── angry/
│   ├── neutral/
│   ├── thumbs_up/
│   ├── peace_sign/
│   ├── open_palm/
│   ├── fist/
│   └── default/           ← fallback if nothing is detected confidently
└── requirements.txt
```

---

## Module responsibilities

### `main.py`
- Open webcam with `cv2.VideoCapture(0)`
- Read frames in a loop
- Pass each frame to MediaPipe for face + hand landmark extraction
- Call `classifier.classify(face_landmarks, hand_landmarks)` → label string
- Call `image_picker.get_image(label)` → image (respects cooldown)
- Call `utils.overlay(frame, reaction_img)` → composited frame
- Show result with `cv2.imshow`
- Exit cleanly on `q` keypress

### `classifier.py`
Exposes one function: `classify(face_landmarks, hand_landmarks) -> str`

**Face expressions** — detect using landmark geometry:

| Label | Detection logic |
|---|---|
| `happy` | Mouth corners (landmarks 61, 291) are higher than mouth centre (13) |
| `surprised` | Eyebrows raised (landmarks 70, 300 high) AND mouth open (13 vs 14 gap large) |
| `angry` | Inner brow corners (landmarks 55, 285) pulled down and inward |
| `neutral` | None of the above thresholds met |

**Hand gestures** — detect using finger extension state:

| Label | Detection logic |
|---|---|
| `thumbs_up` | Only thumb extended upward, all other fingers curled |
| `peace_sign` | Index + middle extended, ring + pinky curled |
| `open_palm` | All five fingers extended |
| `fist` | All fingers curled |

Priority: hand gesture wins over face expression if a hand is detected in frame.

### `image_picker.py`
Exposes: `get_image(label) -> PIL.Image or None`

- Maintains a dict mapping label strings → folder paths
- On a new label, picks a random file from that folder
- Has a cooldown (default 1.5 seconds) — does not pick a new image until the cooldown expires, to prevent flickering
- Falls back to `reactions/default/` if the label folder is empty or missing

### `utils.py`
- `overlay(frame, reaction_img, position="bottom-right", alpha=0.85)` — composites a PIL image onto an OpenCV frame at the given corner, with transparency
- `resize_to_fit(img, max_w, max_h)` — proportional resize
- `Cooldown(seconds)` — simple timer class: `.ready()` returns True and resets if enough time has passed

---

## Landmark reference

MediaPipe uses normalised coordinates (0.0–1.0) relative to frame size. Always multiply by `frame.shape[1]` (width) and `frame.shape[0]` (height) to get pixel positions.

**Key face landmarks:**
- `13` — upper lip centre
- `14` — lower lip centre
- `61` — left mouth corner
- `291` — right mouth corner
- `55` — left inner eyebrow
- `285` — right inner eyebrow
- `70` — left outer eyebrow
- `300` — right outer eyebrow

**Key hand landmarks (both hands):**
- `4` — thumb tip
- `8` — index finger tip
- `12` — middle finger tip
- `16` — ring finger tip
- `20` — pinky tip
- `0` — wrist (base reference)
- `5, 9, 13, 17` — finger base knuckles (MCP joints)

Finger is "extended" when tip y-coordinate is above (lower value than) its base knuckle. Thumb uses x-axis comparison instead.

---

## Detection pipeline (per frame)

```
Frame from webcam
    │
    ├─► MediaPipe FaceMesh  ──► face_landmarks (468 points) ──►┐
    │                                                           ├─► classifier.classify() ──► label
    └─► MediaPipe Hands     ──► hand_landmarks (21 points)  ──►┘
                                                                         │
                                                               image_picker.get_image(label)
                                                                         │
                                                               utils.overlay(frame, img)
                                                                         │
                                                               cv2.imshow("Face Scanner", frame)
```

---

## MediaPipe initialisation

```python
import mediapipe as mp

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
```

Pass frames as RGB (MediaPipe expects RGB, OpenCV gives BGR — convert with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`).

---

## Display layout

```
┌──────────────────────────────────┐
│                                  │
│       Live webcam feed           │
│                                  │
│                       ┌────────┐ │
│                       │ meme / │ │
│  [label text top-left]│reaction│ │
│                       └────────┘ │
└──────────────────────────────────┘
```

- Reaction image: bottom-right corner, max 240×240px, semi-transparent
- Label: top-left, white text with dark outline for readability

---

## Build order

Build in this order so each phase is testable before moving on:

1. **Webcam window** — just open the cam and show the raw feed. Confirm it works.
2. **Landmark overlay** — draw MediaPipe dots on the face and hand. Confirm tracking looks solid.
3. **Classifier (face only)** — implement happy/surprised/angry/neutral, print the label to terminal.
4. **Classifier (hand gestures)** — add hand gesture detection, confirm labels are correct.
5. **Image picker** — load a few test images per folder, confirm random selection and cooldown work.
6. **Overlay compositor** — composite the image onto the frame, confirm it looks right.
7. **Polish** — add confidence thresholds, smooth label transitions, and a clean exit.

---

## Coding conventions

- Use type hints throughout
- Each module is independently importable and testable
- No global state outside `main.py`
- Keep classifier rules as named constants at the top of `classifier.py` so thresholds are easy to tune
- Log detected labels to the terminal during development with a simple `print()` — remove before final version

---

## Potential extensions (out of scope for v1)

- Train a small sklearn classifier on landmark data for more accurate expressions
- Add sound effects per reaction
- Record a GIF of the reaction moment
- Support multiple faces simultaneously