# Seal reference photos

This folder is intentionally empty of images: the repo does not ship third-party or copyrighted artwork.

**Project owner:** drop your own photographs here, one per seal, named exactly:

```
bird.jpg  boar.jpg  dog.jpg  dragon.jpg  hare.jpg  horse.jpg
monkey.jpg  ox.jpg  ram.jpg  rat.jpg  snake.jpg  tiger.jpg  zero.jpg
```

- Shoot them yourself (front view; a side view can be merged into the same picture). About 800 px on the long side, JPEG, plain background.
- Use the SAME canonical variant you decide in `docs/seals.md` (especially **dragon**, which is currently undetectable live and unpinned).
- The app shows `/seals/<seal>.jpg` when present; if the file is missing it shows a placeholder with the text description from `src/content/seals.json`.
- Files here are copied verbatim into the build (`public/`), so add them before `npm run build` / deploying.
- Photos of yourself are fine, but treat them as published once the site is public.
