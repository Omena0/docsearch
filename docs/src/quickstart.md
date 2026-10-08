# Quickstart

## 1. Initialize your docs

```bash
fr-docs init
```

This creates a `docs/` directory with sample configuration and content.

## 2. Edit your configuration

Open `docs/config.json` and customize:
- `project_name`: Your project name
- `project_url`: Your project repository URL
- `sidebar`: Navigation structure

## 3. Write your documentation

Add markdown files to `docs/src/` following the sidebar structure.

## 4. Build your docs

```bash
fr-docs build
```

The output will be in `docs/site/`.

## 5. Preview locally

```bash
cd docs/site && python -m http.server 8000
```

Then open http://localhost:8000

## 6. Deploy

Upload the `docs/site/` directory to your hosting provider (Netlify, Vercel, GitHub Pages, etc.)
