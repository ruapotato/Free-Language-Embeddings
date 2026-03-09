# Project Instructions

## "Push" workflow

When the user says "push", do all of these steps:

1. Stage and commit all changed code and logs (write a descriptive commit message)
2. Run `python web_dashboard.py --export docs/dashboard.html` to regenerate the static dashboard
3. Stage and commit `docs/dashboard.html` (can be part of the same commit or a follow-up)
4. Push to remote (`git push origin main`)
