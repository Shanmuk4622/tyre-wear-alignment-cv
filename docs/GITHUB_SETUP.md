# GitHub Setup

Repo is initialised and committed locally. This file has everything you need to paste into GitHub.

---

## 1. Repo name

```
tyre-wear-alignment-cv
```

Visibility: **Public**

---

## 2. About section (the "Description" field)

GitHub caps this at 350 characters. Use this — **291 chars**:

```
Detailed tyre-wear recognition and single-wheel alignment screening from a low front-mounted camera. Photometric-stereo illumination, high-resolution segmentation, ordinal wear + multi-label patterns, and calibrated geometry with honest uncertainty. VIT-AP capstone research.
```

**Website field:** leave blank for now. Point it at your HF model page once it exists:

```
https://huggingface.co/Shanmuk4622/tyrevision-front-net
```

### Shorter alternative (176 chars) if you prefer terse

```
Detailed tyre-wear recognition + single-wheel alignment screening from one front-facing camera. Photometric stereo, segmentation, ordinal wear, calibrated geometry, uncertainty.
```

---

## 3. Topics (tags)

GitHub allows up to 20. Paste these one at a time in the "Topics" field:

```
computer-vision
deep-learning
pytorch
tire-wear
tyre-inspection
wheel-alignment
semantic-segmentation
photometric-stereo
tread-depth
vehicle-safety
automotive
predictive-maintenance
anomaly-detection
image-processing
opencv
machine-learning
research
capstone-project
dataset
```

The first six matter most — they're what people actually search.

---

## 4. Create and push

### Option A — GitHub CLI (one command)

```bash
cd "D:\Documents\norse\web Applicarion\Tyre"

gh repo create tyre-wear-alignment-cv \
  --public \
  --source=. \
  --remote=origin \
  --push \
  -d "Detailed tyre-wear recognition and single-wheel alignment screening from a low front-mounted camera. Photometric-stereo illumination, high-resolution segmentation, ordinal wear + multi-label patterns, and calibrated geometry with honest uncertainty. VIT-AP capstone research."
```

### Option B — web UI

1. github.com/new
2. Name: `tyre-wear-alignment-cv`, Public
3. **Do not** tick "Add a README", "Add .gitignore", or "Choose a license" — you already have all three. Ticking them creates a conflicting initial commit.
4. Create, then:

```bash
cd "D:\Documents\norse\web Applicarion\Tyre"
git remote add origin https://github.com/<YOUR-USERNAME>/tyre-wear-alignment-cv.git
git push -u origin main
```

> Replace `<YOUR-USERNAME>` with your actual GitHub handle. `CITATION.cff` currently assumes `Shanmuk4622` (your Hugging Face name) — if your GitHub handle differs, fix the `repository-code` line before pushing.

---

## 5. After the first push

- [ ] Paste the About text and topics (§2, §3) — GitHub doesn't take them from the repo
- [ ] Settings → General → Features: **enable Issues**, disable Wikis and Projects for now
- [ ] Create issues from the P0 checklist in `07_ROADMAP.md` — the viewpoint test, the hand-torch photometric test, the gauge study, the parts order. Issues make the timeline visible to your guide.
- [ ] Add topics *before* sharing the link anywhere
- [ ] **Add the other three team members as collaborators** (Settings → Collaborators). Four people, one repo, four subsystem branches.
- [ ] Branch protection on `main`: require a pull request. With four contributors this prevents a bad merge from blocking everyone.

---

## 6. Since you chose public from day one

**The upside you should actually use:** every commit is a public, timestamped record of when you had each idea. That is real scoop protection — better than a private repo, which proves nothing. So:

- **Commit early and often, with descriptive messages.** A commit that says "add wear-geometry cross-check derivation" dated August 2026 is evidence.
- **Tag milestones**: `git tag -a v0.1-review1 -m "Review-1: concept, model stack, literature audit"` then `git push --tags`.
- **Preprint to arXiv** before or alongside journal submission. Until then the public repo establishes priority.

**The one risk to manage:** don't publish the dataset before you've published the paper. Code and docs public is fine. Keep `tyrevision-front` as a **private** HF dataset until your preprint is up, then flip it. `.gitignore` already excludes `data/`, `raw/`, `processed/`, and all model weights, so you can't leak it by accident.

**Privacy reminder:** number plates get blurred and consent forms get signed *before* anything touches a public repo. Retrofitting consent is impossible.

---

## 7. Keeping it current

Add to your Friday routine (`07_ROADMAP.md`):

```bash
cd "D:\Documents\norse\web Applicarion\Tyre"
git add -A
git commit -m "week N: <what changed>"
git push
```
