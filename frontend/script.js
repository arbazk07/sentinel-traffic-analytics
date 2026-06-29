/* ============================================
   script.js
   Handles: source selection (webcam/upload),
   the video feed display, zone-line drawing,
   and polling the live stats endpoint.
   ============================================ */

const webcamBtn = document.getElementById("webcamBtn");
const videoUpload = document.getElementById("videoUpload");
const stopBtn = document.getElementById("stopBtn");
const videoFeed = document.getElementById("videoFeed");
const feedPlaceholder = document.getElementById("feedPlaceholder");
const feedFrame = document.getElementById("feedFrame");
const statusChip = document.getElementById("statusChip");
const fpsChip = document.getElementById("fpsChip");
const drawZoneBtn = document.getElementById("drawZoneBtn");
const zoneHint = document.getElementById("zoneHint");
const zoneCanvas = document.getElementById("zoneCanvas");

const uniqueVisitorsEl = document.getElementById("uniqueVisitors");
const currentlyInFrameEl = document.getElementById("currentlyInFrame");
const zoneCrossingsEl = document.getElementById("zoneCrossings");
const leaderboardList = document.getElementById("leaderboardList");

let statsPollHandle = null;
let isZoneDrawing = false;
let zonePoints = [];

/* ============================================
   SOURCE CONTROLS
   ============================================ */
webcamBtn.addEventListener("click", async () => {
  await startSession({ source: "webcam" });
});

videoUpload.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("video", file);

  setStatus("uploading");
  try {
    const uploadResponse = await fetch("/api/upload", { method: "POST", body: formData });
    const uploadResult = await uploadResponse.json();

    if (!uploadResponse.ok) throw new Error(uploadResult.error || "Upload failed");

    await startSession({ source: "uploaded", filename: uploadResult.filename });
  } catch (err) {
    setStatus("idle");
    alert(`Couldn't start from uploaded video: ${err.message}`);
  }

  videoUpload.value = ""; // reset so the same file can be re-selected later if needed
});

stopBtn.addEventListener("click", async () => {
  await fetch("/api/stop", { method: "POST" });
  stopPollingStats();
  videoFeed.hidden = true;
  feedPlaceholder.hidden = false;
  stopBtn.disabled = true;
  drawZoneBtn.disabled = true;
  setStatus("idle");
  resetStatsDisplay();
});

async function startSession(payload) {
  setStatus("starting");

  try {
    const response = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const result = await response.json();

    if (!response.ok) throw new Error(result.error || "Couldn't start session");

    // Cache-bust the feed URL so the browser doesn't try to reuse a
    // stale connection from a previous session.
    videoFeed.src = `/video_feed?t=${Date.now()}`;
    videoFeed.hidden = false;
    feedPlaceholder.hidden = true;
    stopBtn.disabled = false;
    drawZoneBtn.disabled = false;

    setStatus("live");
    startPollingStats();

  } catch (err) {
    setStatus("idle");
    alert(`Couldn't start: ${err.message}`);
  }
}

function setStatus(state) {
  statusChip.classList.toggle("live", state === "live");
  const labels = {
    idle: "idle",
    starting: "starting…",
    uploading: "uploading…",
    live: "live"
  };
  // statusChip contains a dot <span> followed by a text node ("idle", etc).
  // We update that trailing text node directly rather than touching the
  // dot element, so the dot's own styling/animation stays untouched.
  const textNode = Array.from(statusChip.childNodes).find(node => node.nodeType === Node.TEXT_NODE);
  if (textNode) {
    textNode.textContent = ` ${labels[state] || state}`;
  }
}

/* ============================================
   LIVE STATS POLLING
   ============================================ */
function startPollingStats() {
  stopPollingStats();
  statsPollHandle = setInterval(fetchStats, 1000);
  fetchStats();
}

function stopPollingStats() {
  if (statsPollHandle) {
    clearInterval(statsPollHandle);
    statsPollHandle = null;
  }
}

async function fetchStats() {
  try {
    const response = await fetch("/api/stats");
    const stats = await response.json();

    if (!stats.active) return;

    uniqueVisitorsEl.textContent = stats.unique_visitors_total ?? 0;
    currentlyInFrameEl.textContent = stats.currently_in_frame ?? 0;
    fpsChip.textContent = `${(stats.fps ?? 0).toFixed(1)} fps`;

    zoneCrossingsEl.textContent = stats.zone_configured
      ? `${stats.zone_entries} in / ${stats.zone_exits} out`
      : "— / —";

    renderLeaderboard(stats.dwell_leaderboard || []);

  } catch (err) {
    // A single missed poll isn't worth alarming the user over — the
    // next interval will simply try again.
  }
}

function renderLeaderboard(entries) {
  if (entries.length === 0) {
    leaderboardList.innerHTML = `<p class="leaderboard-empty">No tracked objects yet</p>`;
    return;
  }

  leaderboardList.innerHTML = entries.map(entry => `
    <div class="leaderboard-row ${entry.is_active ? "" : "inactive"}">
      <span class="leaderboard-id">#${entry.track_id}</span>
      <span class="leaderboard-class">${entry.class_name}</span>
      <span class="leaderboard-time">${entry.dwell_seconds.toFixed(1)}s</span>
    </div>
  `).join("");
}

function resetStatsDisplay() {
  uniqueVisitorsEl.textContent = "0";
  currentlyInFrameEl.textContent = "0";
  zoneCrossingsEl.textContent = "— / —";
  fpsChip.textContent = "-- fps";
  leaderboardList.innerHTML = `<p class="leaderboard-empty">No tracked objects yet</p>`;
}

/* ============================================
   ZONE LINE DRAWING
   Click two points on the feed to define a
   counting line; sent to the backend, which
   applies it to live analytics from then on.
   ============================================ */
drawZoneBtn.addEventListener("click", () => {
  isZoneDrawing = !isZoneDrawing;
  zonePoints = [];
  drawZoneBtn.classList.toggle("active", isZoneDrawing);
  zoneCanvas.hidden = !isZoneDrawing;

  if (isZoneDrawing) {
    resizeCanvasToFeed();
    zoneHint.textContent = "Click two points on the feed to set the line";
  } else {
    zoneHint.textContent = "";
    clearCanvas();
  }
});

function resizeCanvasToFeed() {
  const rect = feedFrame.getBoundingClientRect();
  zoneCanvas.width = rect.width;
  zoneCanvas.height = rect.height;
}

function clearCanvas() {
  const ctx = zoneCanvas.getContext("2d");
  ctx.clearRect(0, 0, zoneCanvas.width, zoneCanvas.height);
}

zoneCanvas.addEventListener("click", async (e) => {
  if (!isZoneDrawing) return;

  const rect = zoneCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  zonePoints.push([x, y]);

  drawZonePreview();

  if (zonePoints.length === 2) {
    // Convert canvas-space coordinates to the underlying video's actual
    // pixel coordinates, since the feed is displayed scaled/letterboxed
    // inside the frame — the backend needs real frame coordinates, not
    // on-screen CSS pixel coordinates.
    const scaleX = videoFeed.naturalWidth ? videoFeed.naturalWidth / rect.width : 1;
    const scaleY = videoFeed.naturalHeight ? videoFeed.naturalHeight / rect.height : 1;

    const pointA = [Math.round(zonePoints[0][0] * scaleX), Math.round(zonePoints[0][1] * scaleY)];
    const pointB = [Math.round(zonePoints[1][0] * scaleX), Math.round(zonePoints[1][1] * scaleY)];

    try {
      await fetch("/api/configure_zone", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ point_a: pointA, point_b: pointB })
      });
      zoneHint.textContent = "Counting line set — entries/exits now tracked";
    } catch (err) {
      zoneHint.textContent = "Couldn't set the line — try again";
    }

    isZoneDrawing = false;
    drawZoneBtn.classList.remove("active");
    setTimeout(() => { zoneCanvas.hidden = true; }, 1200);
  }
});

function drawZonePreview() {
  const ctx = zoneCanvas.getContext("2d");
  clearCanvas();

  ctx.fillStyle = "#FFB454";
  zonePoints.forEach(([x, y]) => {
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
  });

  if (zonePoints.length === 2) {
    ctx.strokeStyle = "#FFB454";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.moveTo(zonePoints[0][0], zonePoints[0][1]);
    ctx.lineTo(zonePoints[1][0], zonePoints[1][1]);
    ctx.stroke();
  }
}
