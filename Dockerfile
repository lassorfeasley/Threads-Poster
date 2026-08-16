# Always-on Threads posting scheduler: one long-lived process that ticks the
# posting windows and the metrics poller.
#
# Deliberately not the dashboard. There is no inbound traffic and no clip
# storage — a queued post's clip is uploaded to Supabase on the operator's
# machine and pulled from a signed URL at publish time, so this image needs
# neither a volume nor ffmpeg.
FROM python:3.12-slim

# tzdata is a hard requirement, not a nicety: every posting window is defined in
# America/New_York, and a slim image ships no zoneinfo database for either libc
# or Python's ZoneInfo to read.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Match the container clock to the zone the windows are written in, so log
# timestamps read in the same units as the schedule.
ENV TZ=America/New_York \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# The headless dependency set only — no fastapi, yt-dlp or faster-whisper.
COPY requirements-scheduler.txt .
RUN pip install --no-cache-dir -r requirements-scheduler.txt

# All workspace configs ship in the image; WORKSPACE (set per Fly app in its
# fly.*.toml) picks which one this scheduler serves. Data trees and .env files
# are excluded by .dockerignore — secrets come from `fly secrets set`.
COPY app/ ./app/
COPY workspaces/ ./workspaces/
COPY workspaces.yaml run.py ./

RUN useradd --create-home --uid 1000 scheduler && chown -R scheduler:scheduler /app
USER scheduler

CMD ["python", "run.py", "scheduler", "--loop"]
