# 🚀 Deploying Your Portfolio to GitHub Pages

## Step 1 — Create a GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Name the repo exactly: `indukaBrathnayake.github.io`
   - This special name makes GitHub auto-host it as your personal site
   - Your URL will be: **https://indukaBrathnayake.github.io**
3. Set it to **Public**
4. Click **Create repository**

---

## Step 2 — Upload Your Portfolio Files

### Option A — Upload via GitHub Website (easiest)

1. Open your new repo on GitHub
2. Click **"uploading an existing file"** or **"Add file → Upload files"**
3. Drag and drop the entire contents of the `induka-portfolio/` folder
   - ⚠️ Upload the **contents**, not the folder itself
   - Make sure `index.html` is at the **root level** of the repo
4. Scroll down, add a commit message like `Initial portfolio upload`
5. Click **Commit changes**

### Option B — Using Git (recommended for ongoing updates)

```bash
# In your terminal, go into the portfolio folder
cd induka-portfolio

# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "Initial portfolio upload"

# Link to your GitHub repo
git remote add origin https://github.com/indukaBrathnayake/indukaBrathnayake.github.io.git

# Push to GitHub
git branch -M main
git push -u origin main
```

---

## Step 3 — Enable GitHub Pages

1. Go to your repo on GitHub
2. Click **Settings** (top right tab)
3. In the left sidebar, click **Pages**
4. Under **Source**, select **Deploy from a branch**
5. Set Branch to `main` and folder to `/ (root)`
6. Click **Save**

⏳ Wait **1–2 minutes**, then visit: **https://indukaBrathnayake.github.io**

---

## Step 4 — Add Your CV

Place your CV file at:
```
cv/Induka_Rathnayake_CV.pdf
```

The download button is already linked to this path. Just drop your PDF there!

---

## Step 5 — Update Your Weekly FYP Log

Open `projects/final-year-project.html` in any text editor. Find the `weekData` array and edit each week's entry:

```javascript
{
  week: 1,
  title: "Your actual week title",
  date: "Jan 15 – Jan 22, 2025",   // add actual dates
  status: "completed",              // "completed" | "current" | "upcoming"
  summary: "What you achieved this week...",
  activities: "Tasks you carried out...",
  findings: "Key results or observations...",
  files: [
    { name: "Week 1 Report.pdf", url: "fyp-files/week1-report.pdf" }
  ]
}
```

Upload any files to `projects/fyp-files/` and reference them in the `files` array.

---

## Step 6 — Set Up the Contact Form (Optional)

1. Go to [formspree.io](https://formspree.io) and create a free account
2. Create a new form and copy the form ID (looks like `xeqwabcd`)
3. In `index.html`, replace:
   ```
   https://formspree.io/f/YOUR_FORM_ID
   ```
   with your actual Formspree URL

---

## Updating the Site Later

Every time you make changes:

```bash
git add .
git commit -m "Update week 3 progress"
git push
```

Changes go live within ~30 seconds.

---

## File Structure Reference

```
indukaBrathnayake.github.io/
├── index.html               ← Main portfolio page
├── css/
│   ├── style.css            ← Template styles
│   ├── custom.css           ← YOUR customisations
│   └── ...
├── js/
│   └── ...
├── fonts/
│   └── ...
├── images/
│   ├── face.png             ← Replace with your photo!
│   └── ...
├── cv/
│   └── Induka_Rathnayake_CV.pdf  ← Add your CV here
└── projects/
    ├── final-year-project.html
    ├── mini-solar.html
    ├── noaa.html
    └── fyp-files/           ← Upload FYP weekly files here
        ├── week1-report.pdf
        └── ...
```

---

## Tips

- **Replace `images/face.png`** with your actual photo (same filename, or update `index.html`)
- **Add project images** named `proj_fyp.jpg`, `proj_solar.jpg`, `proj_noaa.jpg` to the `images/` folder — the cards will automatically use them instead of the colour placeholders
- **Custom domain**: If you want `www.indukarathnayake.com`, buy a domain and set it in repo Settings → Pages → Custom domain

---

*Your live portfolio URL: https://indukaBrathnayake.github.io*
