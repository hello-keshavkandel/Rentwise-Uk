# Pushing RentWise UK to GitHub (macOS + VSCode)

You push **only this `Rentwise-UK` folder** as the repository — not the whole
`Claude_Dissertation2026` folder (that contains private specs and emails).

Your `.gitignore` already excludes the data files, `.venv`, and caches, so only
code and documentation get published. Good.

---

## Option A — Easiest: VSCode "Publish to Branch" (GUI, recommended)

1. In VSCode: **File → Open Folder →** select the `Rentwise-UK` folder (open this
   folder as the project root, not its parent).
2. Click the **Source Control** icon in the left bar (branch-looking icon), or press
   `Ctrl+Shift+G`.
3. Click **Initialize Repository**.
4. Type a commit message in the box, e.g. `Initial commit: setup + data cleaning`,
   then click the **✓ Commit** button. If it asks "stage all changes?", say Yes.
5. Click **Publish Branch**. VSCode asks: **public or private** repository — choose
   **Private** for now (you can make it public later for the report link).
6. The first time, a browser window opens to **sign in to GitHub** and authorise
   VSCode. Approve it. VSCode remembers this afterwards.
7. Done — your code is now on GitHub. Click the link VSCode shows to view it.

---

## Option B — Command line (if you prefer the terminal)

First create an EMPTY repo on github.com:
1. Go to https://github.com/new
2. Repository name: `rentwise-uk`  ·  set **Private**  ·  do NOT add a README,
   .gitignore or licence (we already have them).  Click **Create repository**.
3. Copy the repo URL (e.g. `https://github.com/YOUR-USERNAME/rentwise-uk.git`).

Then in the VSCode terminal, inside the `Rentwise-UK` folder:
```bash
git init -b main                       # start a git repo on a 'main' branch
git add -A                             # stage all (gitignored files are skipped)
git commit -m "Initial commit: setup + data cleaning"
git remote add origin https://github.com/YOUR-USERNAME/rentwise-uk.git
git push -u origin main
```

### Authentication note (important)
GitHub no longer accepts your account password on the command line. When it asks
for a password, paste a **Personal Access Token** instead:
1. https://github.com/settings/tokens → **Generate new token (classic)**
2. Tick the **repo** scope, generate, and copy the token.
3. Use your GitHub **username** and paste the **token** as the password.
(Option A avoids this entirely by signing in through the browser.)

---

## Everyday workflow after the first push
Each time you change code:
```bash
git add -A
git commit -m "describe what changed, e.g. add EDA (Phase 3)"
git push
```
Or in VSCode: type a message, click **Commit**, then **Sync Changes**.

## What NOT to commit
- Real ONS/HM Land Registry data files (they can be large and are re-downloadable).
  The `.gitignore` already blocks `data/raw/*` and `data/processed/*`.
- Your `.venv` folder (also already ignored).
