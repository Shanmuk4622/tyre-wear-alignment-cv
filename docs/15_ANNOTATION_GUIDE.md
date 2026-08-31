# 15 — Annotation Guide (Windows)

> **Tool: labelme.** Runs entirely on your own Windows machine. No account, no login, no task limits, no uploading. Segment Anything (SAM2) is built in and runs offline.
>
> **Watch first (~10 min):** [How to Install and Use LabelMe — Step-by-Step Tutorial for Beginners](https://www.youtube.com/watch?v=PtUO_H3DEc8)
> Then come back here. Everything tyre-specific is in Part C.

> **Why not CVAT Online:** the free tier caps how many tasks you can create, and we hit that cap. labelme has no limits because nothing leaves your computer.

---

## 0. Read this before anything else

**You are annotating 418 images. Not 4,598.**

4,180 of the dataset's images are *automatic transforms* of the other 418 — the same photo rotated, cropped or flipped, with the exact parameters recorded. We annotate the 418 originals and a script replays those transforms on the masks.

**Annotate 418 → get 4,598.** Do not touch the `augmented` folder.

**You will not draw outlines by hand.** You click once on the tyre, SAM produces a mask that is usually 90–95% right, and you correct it. **~25–40 seconds per image.**

| | |
|---|---|
| Images | **418** |
| People | **1 — you** |
| **Total time** | **~3.5–4 hours**, best split over 3–4 sittings |
| Plus | ~20 min consistency pass at the end |
| Tool | labelme — local, offline, free |
| Internet needed | Only once, to download the AI model (~150 MB) |

> **Do not try to do this in one sitting.** Your accuracy drops after about an
> hour, and on this job accuracy *is* the deliverable. Four sittings of an hour,
> one session-group at a time, is the right shape. §D explains how to measure
> whether your judgement stayed consistent.

---

# PART A — Install labelme on Windows (~5 minutes)

## A1. Open Anaconda Prompt

Press the **Windows key**, type `anaconda prompt`, click **Anaconda Prompt**.

> Use Anaconda Prompt, **not** the regular Command Prompt or PowerShell. `conda activate` will not work in those unless you have configured them.

## A2. Install

Type these two lines, pressing **Enter** after each:

```
conda activate cv_conda
pip install labelme
```

Installation takes 1–3 minutes. You will see a lot of scrolling text. When you get the prompt back and see `Successfully installed labelme-...`, it worked.

## A3. Check it opens

```
labelme
```

A window titled **labelme** should appear. Close it for now.

> **First launch can take 20–30 seconds.** That is normal.

## A4. If something went wrong

| Error | Fix |
|---|---|
| `'conda' is not recognized` | You are in the wrong terminal. Use **Anaconda Prompt** |
| `EnvironmentNameNotFound: cv_conda` | Run `conda env list` to see your environments and use the right name, or create it: `conda create -n cv_conda python=3.11 -y` |
| Errors mentioning **PyQt5** or **Qt** | `pip uninstall -y opencv-python` then `pip install opencv-python-headless` then `pip install --force-reinstall labelme` |
| `ImportError: DLL load failed` | Install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe), reboot, try again |
| Window opens then closes instantly | Run from Anaconda Prompt (not by double-clicking) so you can read the error message |
| Anything else | `pip install --upgrade labelme` and try again |

---

# PART B — Build the package (~5 minutes)

## B1. Run the prep script

In Anaconda Prompt:

```
conda activate cv_conda
cd /d "D:\Documents\norse\web Applicarion\Tyre"
python scripts\prepare_annotation_batches.py --final "D:\Dataset Download\Tire Dataset Prepared\FINAL" --out "D:\Dataset Download\Tire Dataset Prepared\annotation_work"
```

*(The last line is one line. `cd /d` is needed to change drive **and** folder on Windows.)*

You get:

```
annotation_work\
├── batch_ALL\          418 images, ordered by session
├── batch_SELFCHECK\     30 images — a copy, for the consistency pass at the end
├── labels.txt           the fixed label list
├── selfcheck_ids.txt    which 30 they are
└── filename_map.csv     KEEP THIS — the import script needs it
```

The script prints the session order and how many images each session has. **Work
through them in that order** — one tyre at a time, so your boundary judgement
stays consistent within a tyre.

## B2. Check `labels.txt`

Open `annotation_work\labels.txt` in Notepad. It must contain exactly this:

```
__ignore__
_background_
tyre
tread
marking
damage
```

> The first two lines are labelme's internal entries — leave them. Loading this
> file means you **pick** labels from a fixed list instead of typing them, so
> `Tread` or `tyre ` (trailing space) can never happen. That one detail saves an
> afternoon of cleanup.

---

# PART C — Annotating

## C1. Start labelme

In Anaconda Prompt, one line (the prep script prints this exact command for you):

```
conda activate cv_conda
labelme "D:\Dataset Download\Tire Dataset Prepared\annotation_work\batch_ALL" --labels "D:\Dataset Download\Tire Dataset Prepared\annotation_work\labels.txt" --output "D:\Dataset Download\Tire Dataset Prepared\annotation_work\ann_pass1" --nodata --autosave
```

**What each flag does:**

| Flag | Why |
|---|---|
| `--labels labels.txt` | Fixed label list — no typos possible |
| `--output ..._annotations` | Keeps the JSON files in their own folder, away from the images |
| `--nodata` | Does not embed a copy of the image in every JSON. Files stay ~20 KB instead of ~2 MB |
| `--autosave` | Saves as you go |

> **Make a shortcut — you will run this a dozen times.** Paste the command into
> Notepad, save as `start_annotating.bat` on your Desktop, and double-click it
> each sitting. Save as **All Files**, not `.txt`.
>
> labelme reopens on the first image every time. Press **`D`** to skip forward to
> where you left off, or use **File → Open Next** — the images you have already
> done show a filled dot in the file list on the right.

## C2. Turn on the AI tool (once per session)

1. Look at the **toolbar** across the top of the labelme window
2. Find the **AI-Assisted Annotation** widget — a dropdown showing a model name
3. Leave it on **Sam2 (balanced)** — the default, and the right choice
4. Next to it is an **output shape** dropdown → set it to **Polygon**

> **Polygon, not Mask.** A polygon can be edited point by point afterwards. A raw mask cannot.

**The very first time you use it, labelme downloads the model (~150 MB).** The first click will take 30–60 seconds and look frozen. It is not. After that it is fast and completely offline.

## C3. Annotating one image — click by click

This is the loop, 418 times. Read it slowly once; after about ten images it
becomes automatic and you stop thinking about the mechanics.

### Step 1 — the `tyre` mask

1. Click **AI-Points** in the toolbar
2. **Left-click once in the middle of the tyre**
   - After a moment an outline appears around the tyre
3. **Fix it:**
   - **Left-click** on any part of the tyre it *missed* → adds that area
   - **`Shift` + left-click** on anything it wrongly *included* (road, wall, car body) → removes it
   - Two or three corrections is normal. Ten means start over: press `Esc` and click somewhere more central
4. Press **`Enter`** to accept
5. A small box pops up asking for the label → click **`tyre`** in the list → **OK**

### Step 2 — the `tread` mask

1. Click **AI-Points** again
2. **Left-click in the middle of the grooved band** — the flat face with grooves running around it
3. SAM will usually grab the whole tyre including the shoulders. Expected.
4. **`Shift` + left-click on each shoulder** — the curved parts on the left and right where the surface turns away from you. Usually 2–4 shift-clicks.
5. Press **`Enter`**
6. Choose **`tread`** → **OK**

### Where exactly does `tread` stop?

**This is the only real judgement call in the whole job, and it is the one that matters most** — every attention metric in the study is measured against this mask.

> ### The rule
> **`tread` ends at the outermost full-depth groove that runs all the way around the tyre.**
>
> Look at the grooves. In the middle they are wide, dark and clearly open. Toward the edges they get shorter and angle away from you. **The last groove that still looks full-depth and open is the boundary.** Past that is shoulder — exclude it.

**If you genuinely cannot tell, include LESS rather than more.** A slightly small tread mask makes our measurements slightly pessimistic, which is the safe direction. A too-large one silently inflates every result.

```
        ← shoulder →│←──────── TREAD ─────────→│← shoulder →
                    │  ║   ║   ║   ║   ║   ║   │
    curved,         │  ║   ║   ║   ║   ║   ║   │      curved,
    grooves         │  full-depth grooves       │      grooves
    foreshortened   │                           │      foreshortened
    EXCLUDE         │  INCLUDE                  │      EXCLUDE
```

### Step 3 — `marking`, **only if visible**

Factory paint stripes (blue, green, yellow, orange) and white moulded lettering. **New tyres almost always have them. Very worn tyres almost never do.**

These are small and thin, so use a manual polygon:

1. Press **`Ctrl` + `N`** (polygon mode)
2. **Click around the outline** of the stripe or the lettering — 6–10 points is plenty
3. Press **`Enter`** to close it
4. Choose **`marking`** → **OK**

> **Why this small mask matters more than it looks.** A model can score 95% by learning "blue stripe = new tyre" without ever looking at the tread. This mask is how we *catch* that. Be accurate here — a little work protecting a large claim.

**No marking visible? Skip this step.** Do not invent one.

### Step 4 — `damage`, **only if visible**

Cuts, missing chunks, exposed cord, a stone wedged in a groove.

1. **`Ctrl` + `N`**, click around it, **`Enter`**
2. Choose **`damage`** → **OK**

**Most images have none. Skip it.**

### Step 5 — next image

1. Press **`Ctrl` + `S`** (autosave is on, but do this anyway)
2. Check the **Polygon Labels** panel on the right — you should see at least `tyre` and `tread`
3. Press **`D`** for the next image

**That is one image.**

## C4. The loop, condensed

```
AI-Points → click tyre centre → Shift+click mistakes → Enter → tyre
AI-Points → click tread centre → Shift+click both shoulders → Enter → tread
Ctrl+N → outline stripe (if any) → Enter → marking
Ctrl+N → outline damage (if any) → Enter → damage
Ctrl+S  →  D
```

## C5. Keyboard shortcuts

| Key | Does |
|---|---|
| **`D`** | **Next image** |
| **`A`** | **Previous image** |
| **`Enter`** or `Space` | Finish the current shape |
| **`Ctrl` + `S`** | Save |
| `Ctrl` + `N` | Polygon mode (manual) |
| `Ctrl` + `J` | Edit mode — select, move, resize |
| `Ctrl` + `Z` | Undo |
| `Delete` | Delete the selected shape |
| `Esc` | Cancel the shape you are drawing |
| `Ctrl` + scroll | Zoom |

> **`D` is next, `A` is previous.** Not the arrow keys. Everyone gets this wrong for the first ten minutes.

## C6. Two settings worth changing first

- **View → Show Labels** — turn on so you can see which shape is which
- Zoom so the tyre fills most of the window (`Ctrl` + scroll)

---

# PART D — The consistency pass (~20 min, at the END)

**Do this after all 418 are done. Do not skip it.**

## Why it matters more when you work alone

With four annotators you measure whether *they* agree with each other. Working
alone, the risk is different and less obvious: **your own judgement drifts.**

By image 300 your eye is much better at spotting where the tread ends than it
was at image 10. That is a good thing — but it means the first images and the
last images were labelled by, effectively, two different people. Nothing warns
you this has happened.

So we measure it: re-annotate 30 images you did earlier, without looking at your
first attempt, and compare.

## D1. Second pass

```
labelme "D:\...\annotation_work\batch_SELFCHECK" --labels "D:\...\annotation_work\labels.txt" --output "D:\...\annotation_work\ann_pass2" --nodata --autosave
```

**Do not open your first-pass files first.** The whole point is an independent
judgement. Just annotate the 30 as if you had not seen them.

## D2. Compare

```
conda activate cv_conda
cd /d "D:\Documents\norse\web Applicarion\Tyre"
python scripts\annotation_agreement.py --mode self --pass1 "D:\...\annotation_work\ann_pass1" --pass2 "D:\...\annotation_work\ann_pass2"
```

| Metric | Must be at least |
|---|---|
| `tyre` IoU | 0.95 |
| **`tread` IoU** | **0.90** |
| `marking` IoU | 0.80 |
| `damage` presence agreement | 0.70 |

## D3. If `tread` comes back below 0.90

Your boundary judgement drifted. Not a disaster, and it is exactly what this
check exists to catch.

**Your later work is almost certainly the better work** — your eye improved.
So:

1. Re-read Part C, Step 2 and fix the rule in your head
2. **Re-annotate the first ~40 images of `batch_ALL`** — those were done before
   you had it internalised
3. Re-run the comparison

That is 20 minutes of work that meaningfully improves everything downstream.

## D4. Report the number

Put the final `tread` IoU in `PROGRESS.md` and in the paper. Self-consistency is
a legitimate, reportable quality measure, and **almost no student project
measures it at all**. An examiner who asks "how do you know your labels are
reliable?" gets a number instead of a shrug.

---

# PART E — Finishing up

## E1. Check the count first

```
dir "D:\...\annotation_work\ann_pass1\*.json"
```

The number of JSON files must be **418**. If it is short, you missed some —
labelme only writes a file for images you actually drew on. Reopen and press
`D` through the batch looking for images with no shapes in the right-hand panel.

## E2. Import, validate, propagate

Three commands, each one line:

```
conda activate cv_conda
cd /d "D:\Documents\norse\web Applicarion\Tyre"

python scripts\import_annotations.py --exports "D:\...\annotation_work\ann_pass1" --map "D:\...\annotation_work\filename_map.csv" --out "D:\Dataset Download\Tire Dataset Prepared\annotations" --format labelme

python scripts\validate_annotations.py --annotations "D:\Dataset Download\Tire Dataset Prepared\annotations" --final "D:\Dataset Download\Tire Dataset Prepared\FINAL"

python scripts\propagate_annotations.py --annotations "D:\Dataset Download\Tire Dataset Prepared\annotations" --final "D:\Dataset Download\Tire Dataset Prepared\FINAL"
```

`validate_annotations.py` lists any image missing a `tyre` or `tread` mask, any
size mismatch, and any tread mask suspiciously close to the whole tyre (which
usually means the shoulders were not excluded). **Fix what it reports before
running step 3.**

## E3. Read the verification, then LOOK at the overlays

Step 3 now checks its own work before it claims success. Each propagated mask
is scored against three deliberately wrong versions of **itself**, on the same
image: shifted 6% sideways, mirrored, and swapped with another image's mask. A
correct mask beats all three by a wide margin.

```
=== verification ===   (n=200)
  propagated mask, as written      32.89
  control: shift 6%                 19.30
  control: mirrored                 12.37
  control: swapped                  13.45
  margin over the best control    +13.59   (want > +5.00)

  PASS.
```

The script exits non-zero on FAIL, so it cannot pass silently.

> **This check exists because it caught a real bug.** The first version of the
> propagation script matched operations by substring and looked for parameters
> under key names the dataset does not use (`box`, `angle`, where the trace
> records `crop_box`, `degrees`). Both lookups quietly found nothing, so the
> **crop and the rotation were skipped on all 4,180 files** — which were
> written anyway, correctly sized, with legal class values, and looked
> completely normal. They scored 16.1 against a swap control of 9.8: barely
> better than a mask from a different photograph.
>
> The script now fails loudly on any operation it cannot classify, rather than
> ignoring it.

Then also write 24 overlay images to `annotations\audit\propagation_check\`.
**Open the folder in File Explorer and look at them.** The numbers catch gross
misalignment; your eyes catch the subtle kind. Thirty seconds of looking.

## E4. The real test: NBT1

`notebooks\NBT1_Annotation_Test.ipynb` trains a segmentation model on the 418
hand-drawn masks alone, then scores it on held-out tyres against **both** their
hand-drawn masks and the propagated masks of their derivatives. The model has
never seen a propagated mask, so if the two scores agree the replay put them in
the right place — and if the second collapses toward the shuffled-label
control, it did not.

~25 minutes on a T4. Run it after any change to the annotations.

**Real-run result (2026-08-30): PASS.** NBT1 found that the supplied propagated
masks were not aligned and rebuilt all 4,180 from the 418 clean masks and
transform traces. The masks actually tested had fingerprint
`085acfb8fb83c531`. Clean IoU was 0.9780, propagated IoU 0.9747, and their ratio
0.9966; all seven revision-specific artifacts were verified public. The
ignored data-loader cleanup messages around epoch 18 were benign and are fixed
by using no worker subprocesses for arrays already in memory.

**Nothing in it is gated on a version string.** It prints
`ANNOTATION_VERSION.json` for information and then ignores it. A label is not
evidence: it can say `v2` over broken masks, and `v1` over fixed ones. Part B
measures the masks in three minutes and that measurement is the gate.

### ⚠ Kaggle pins a dataset *version* per notebook

This costs people a day, reliably, so it is worth stating plainly:

> Adding a dataset to a notebook attaches **one specific version** of it.
> Uploading a new version does **not** move existing notebooks onto it. They
> keep reading the version they were attached to, silently, with nothing on
> screen to say so.

That is why NBT1 prints fingerprints at startup. The quick eight-file digest
below identifies the supplied local package; the final `MASK_FINGERPRINT`
printed by NBT1 identifies the complete mask set it actually used after any
self-healing. Compare the quick digest with your own folder:

```
python -c "import hashlib,pathlib;p=sorted(pathlib.Path(r'D:/Dataset Download/Tire Dataset Prepared/annotations/propagated/masks').glob('*.png'))[:8];print(hashlib.sha256(b''.join(f.read_bytes() for f in p)).hexdigest()[:16])"
```

Same digest → Kaggle has your current masks. Different → **Input panel → the
dataset → version dropdown → newest.** The session restarts and you are reading
the right data.

For the supplied `annotations/` folder the eight-file digest is
**`301de19631fc026b`**. For the self-healed mask set used by the successful real
run, NBT1's complete-set fingerprint is **`085acfb8fb83c531`**. They hash
different scopes and are not expected to match.

---

# PART F — Where files live

Parallel to the images, **never inside `FINAL\`** — that package is immutable.

```
Tire Dataset Prepared\
├── FINAL\                        ← DO NOT TOUCH
└── annotations\
    ├── ANNOTATION_VERSION.json
    ├── clean\
    │   ├── masks\<image_id>.png       0=bg 1=tyre 2=tread 3=marking 4=damage
    │   ├── polygons\<image_id>.json   the labelme files, editable
    │   └── boxes\<image_id>.txt       YOLO format, derived from masks
    ├── propagated\masks\              generated — never hand-edit
    ├── coco\instances_clean.json
    └── audit\
        ├── agreement_report.json      self-consistency, pass1 vs pass2
        ├── validation_report.json
        └── propagation_check\         ← LOOK AT THESE
```

**`v1` is frozen once experiments start.** Corrections create `v2` with a changelog — the same discipline the dataset package applies to itself.

---

# PART G — Troubleshooting

| Problem | Fix |
|---|---|
| First AI click seems frozen | It is downloading the model (~150 MB). Wait 60 s. Only happens once |
| AI click takes 5–10 s every time | Normal on a CPU-only laptop. If you have an NVIDIA GPU it will be faster. Consider switching the model dropdown to an **EfficientSam (speed)** variant |
| SAM grabs the road as well as the tyre | **`Shift` + click** on the road. Two or three usually fixes it |
| SAM misses the dark lower part of the tyre | **Left-click** directly on the dark part to add it |
| I cannot edit the shape afterwards | Output shape was set to **Mask**. Delete it, set the dropdown to **Polygon**, redo |
| The label popup does not list my labels | You forgot `--labels labels.txt` in the command. Close and restart with it |
| Wrong label picked | `Ctrl`+`J` (edit mode), double-click the shape, change the label |
| `D` does nothing | Click once on the image first to give it focus |
| Drew something wrong | Click it in the right panel, press `Delete`. Or `Ctrl`+`Z` |
| Where did my work go? | The `--output` folder, one `.json` per image. Not next to the images |
| labelme is slow / laggy | Close other programs. Large images are heavy — this is normal on 1152×1536 |
| Tyre runs off the edge of the frame | Annotate what is visible. Do not guess beyond the frame. Note the filename and send the list to Shanmukesh — these get flagged `truncated` |
| Two tyres in one image | Annotate only the large, central, in-focus one |
| Too blurry to find the tread edge | Do your best, note the filename. Genuinely unusable ones get excluded and the exclusion is reported |
| Accidentally closed labelme | Everything is autosaved. Restart with the same command and press `D` until you reach where you were |
| Lost my place in 418 images | Images already done show a filled dot in the file list on the right. Or sort `ann_pass1` by date modified and read the last filename |
| Getting tired / sloppy | **Stop.** Accuracy is the deliverable here. Come back tomorrow — the consistency pass in Part D will tell you honestly whether a bad session crept in |
| Realised halfway that my tread boundary was wrong | Finish the batch with the *correct* rule, then re-annotate the early images. Part D §D3 |

---

# PART H — Checklist

**Before starting**

- [ ] `pip install labelme` succeeded and `labelme` opens
- [ ] `prepare_annotation_batches.py` run; `batch_ALL`, `batch_SELFCHECK`, `labels.txt`, `filename_map.csv` exist
- [ ] Watched the [video](https://www.youtube.com/watch?v=PtUO_H3DEc8)
- [ ] Read Part C twice — especially the tread-boundary rule
- [ ] Made `start_annotating.bat` so you do not retype the command
- [ ] Output shape dropdown set to **Polygon**, not Mask

**Each sitting**

- [ ] Work through one session-group at a time, in the printed order
- [ ] Every image gets a `tyre` mask **and** a `tread` mask
- [ ] `marking` wherever a paint stripe or lettering is visible
- [ ] `damage` wherever damage is visible
- [ ] `Ctrl`+`S` every ten images or so
- [ ] Stop after about an hour

**When all 418 are done**

- [ ] JSON count is 418
- [ ] Second pass on `batch_SELFCHECK` into `ann_pass2` — **without looking at pass 1**
- [ ] `annotation_agreement.py --mode self` run
- [ ] **`tread` IoU cleared 0.90** (if not, re-annotate the first ~40 — §D3)
- [ ] `import_annotations.py` run
- [ ] `validate_annotations.py` passes, problems fixed
- [ ] `propagate_annotations.py` run
- [ ] **Propagation overlays actually looked at**
- [ ] Self-consistency number recorded in `PROGRESS.md`
- [ ] `annotations/` pushed to Hugging Face

---

## Time budget

| Step | Time |
|---|---|
| Install labelme | 5 min |
| Build the package | 5 min |
| Video + reading Part C | 20 min |
| **Annotate 418 images** | **~3.5–4 h, split over 3–4 sittings** |
| Consistency pass (30 images) | 20 min |
| Import + validate + propagate | 15 min |

**About five hours of your time in total**, and it removes the largest
methodological weakness in the study. Every attention metric currently rests on
masks nobody has checked. After this they rest on ground truth, with a measured
consistency figure attached.

### A sensible schedule

| Sitting | What | Images |
|---|---|---|
| 1 | Install, read, then `new_tire_001` + `100000_plus_002` | ~69 |
| 2 | `000100_005000_001` + `_002` | ~102 |
| 3 | `040000_001`, `070000_001`, `090000_001` | ~97 |
| 4 | the five remaining `100000_plus` sessions | ~150 |
| 5 | consistency pass + import + propagate | 30 |

One session-group at a time keeps your boundary judgement consistent within a
tyre, which is where consistency matters most.

## If you would rather use something else

| Tool | Verdict |
|---|---|
| **labelme** | ✅ **What this guide uses.** `pip install`, offline, no limits, SAM2 built in |
| CVAT Online | ❌ Free tier caps the number of tasks — we hit it |
| CVAT self-hosted | ○ No limits, but needs Docker Desktop. More setup than a one-person job needs |
| Label Studio | ○ Easy to install (`pip install label-studio`), but SAM needs a separate ML backend configured |
| X-AnyLabeling | ○ A labelme fork with more SAM variants. Worth trying if labelme's AI is slow on your machine |
| Roboflow | ❌ Free tier also has limits, and it uploads your images |

---

## Reference

- **Video:** [How to Install and Use LabelMe — Step-by-Step Tutorial for Beginners](https://www.youtube.com/watch?v=PtUO_H3DEc8)
- [labelme annotation basics](https://labelme.io/docs/annotation-basics) — shape types, every shortcut
- [labelme installation docs](https://labelme.io/docs/installation)
- [labelme on GitHub](https://github.com/wkentaro/labelme)
