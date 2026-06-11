// AgriSense — Frontend Application
// Leaflet map + drawing + SAM interaction + job polling + chat

// ============================
// MAP SETUP
// ============================

const map = L.map("map").setView([48.0, 68.0], 5); // Kazakhstan overview

const baseLayers = {
    "Satellite": L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        { attribution: "Esri", maxZoom: 18 }
    ),
    "OpenStreetMap": L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        { attribution: "OSM", maxZoom: 19 }
    ),
};
baseLayers["Satellite"].addTo(map);

// Result layers
const previewLayer = L.layerGroup().addTo(map);
const ndviLayer = L.layerGroup().addTo(map);
const samLayer = L.layerGroup().addTo(map);
const samPointsLayer = L.layerGroup().addTo(map);
const cropLayer = L.layerGroup().addTo(map);
const seasonLayer = L.layerGroup().addTo(map);

const overlayLayers = {
    "Satellite Preview": previewLayer,
    "NDVI": ndviLayer,
    "Crop Map": cropLayer,
    "Season (NDVI)": seasonLayer,
    "SAM Segments": samLayer,
    "SAM Points": samPointsLayer,
};

L.control.layers(baseLayers, overlayLayers, { collapsed: false }).addTo(map);

// ============================
// DRAWING
// ============================

const drawnItems = new L.FeatureGroup().addTo(map);
const drawControl = new L.Control.Draw({
    draw: {
        rectangle: { shapeOptions: { color: "#58a6ff", weight: 2, fillOpacity: 0.1 } },
        polygon: false,
        circle: false,
        marker: false,
        polyline: false,
        circlemarker: false,
    },
    edit: { featureGroup: drawnItems },
});
map.addControl(drawControl);

let currentBbox = null;
let samMode = false;
let samPoints = [];
let samLabels = [];

// Clear all analysis overlays + UI so a new selection starts clean.
function clearResults() {
    [previewLayer, ndviLayer, cropLayer, seasonLayer, samLayer, samPointsLayer]
        .forEach(l => l.clearLayers());
    samMode = false;
    samPoints = [];
    samLabels = [];
    document.getElementById("sam-controls").classList.add("hidden");
    document.getElementById("season-slider").classList.add("hidden");
}

map.on(L.Draw.Event.CREATED, function (e) {
    clearResults();
    drawnItems.clearLayers();           // only ever one selection rectangle
    drawnItems.addLayer(e.layer);
    const bounds = e.layer.getBounds();
    currentBbox = [
        bounds.getWest(), bounds.getSouth(),
        bounds.getEast(), bounds.getNorth(),
    ];
    updateBboxInfo();
});

function setBbox(bbox) {
    currentBbox = bbox;
    clearResults();
    drawnItems.clearLayers();
    const bounds = [[bbox[1], bbox[0]], [bbox[3], bbox[2]]];
    console.log("Leaflet bounds:", bounds);
    const rect = L.rectangle(bounds, {
        color: "#58a6ff",
        weight: 3,
        fillOpacity: 0.15,
        dashArray: "8, 4",
        interactive: false,
    });
    drawnItems.addLayer(rect);
    map.fitBounds(rect.getBounds(), { padding: [50, 50] });
    updateBboxInfo();
}

function updateBboxInfo() {
    const el = document.getElementById("bbox-info");
    if (currentBbox) {
        el.textContent = `bbox: [${currentBbox.map(v => v.toFixed(4)).join(", ")}]`;
        el.classList.remove("hidden");
    } else {
        el.classList.add("hidden");
    }
}

// ============================
// PRESETS
// ============================

fetch("/api/presets")
    .then(r => r.json())
    .then(data => {
        const grid = document.querySelector(".preset-grid");
        data.regions.forEach(region => {
            const btn = document.createElement("button");
            btn.className = "preset-btn";
            btn.textContent = region.name;
            btn.addEventListener("click", () => setBbox(region.bbox));
            grid.appendChild(btn);
        });
    });

// ============================
// TOOL BUTTONS
// ============================

function getDateRanges() {
    return [
        document.getElementById("date-spring-start").value + "/" + document.getElementById("date-spring-end").value,
        document.getElementById("date-peak-start").value + "/" + document.getElementById("date-peak-end").value,
        document.getElementById("date-late-start").value + "/" + document.getElementById("date-late-end").value,
    ];
}

document.getElementById("btn-preview").addEventListener("click", () => {
    if (!currentBbox) return alert("Draw a rectangle on the map first.");
    runJob("satellite-preview", { bbox: currentBbox, date_ranges: getDateRanges() });
});

document.getElementById("btn-ndvi").addEventListener("click", () => {
    if (!currentBbox) return alert("Draw a rectangle on the map first.");
    runJob("ndvi", { bbox: currentBbox, date_ranges: getDateRanges() });
});

document.getElementById("btn-timeseries").addEventListener("click", () => {
    if (!currentBbox) return alert("Draw a rectangle on the map first.");
    runJob("timeseries", { bbox: currentBbox });
});

document.getElementById("btn-cropmap").addEventListener("click", () => {
    if (!currentBbox) return alert("Draw a rectangle on the map first.");
    runJob("cropmap", { bbox: currentBbox });
});

document.getElementById("btn-sam-load").addEventListener("click", () => {
    if (!currentBbox) return alert("Draw a rectangle on the map first.");
    const dateRanges = getDateRanges();
    runJob("sam/load", {
        bbox: currentBbox,
        date_range: dateRanges[1], // peak season
    });
});

document.getElementById("btn-sam-clear").addEventListener("click", () => {
    samPoints = [];
    samLabels = [];
    samPointsLayer.clearLayers();
    samLayer.clearLayers();
});

// ============================
// SAM CLICK HANDLER
// ============================

map.on("click", async function (e) {
    if (!samMode) return;

    const clickType = document.querySelector('input[name="sam-click"]:checked').value;
    const label = clickType === "fg" ? 1 : 0;

    samPoints.push([e.latlng.lng, e.latlng.lat]);
    samLabels.push(label);

    // Show point marker
    const icon = L.divIcon({
        className: label === 1 ? "sam-point-fg" : "sam-point-bg",
        iconSize: [14, 14],
    });
    L.marker(e.latlng, { icon }).addTo(samPointsLayer);

    // Run prediction
    try {
        const resp = await fetch("/api/sam/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ points: samPoints, labels: samLabels }),
        });
        const data = await resp.json();

        if (data.geojson) {
            samLayer.clearLayers();
            L.geoJSON(data.geojson, {
                style: {
                    color: "#3fb950",
                    weight: 2,
                    fillColor: "#3fb950",
                    fillOpacity: 0.25,
                },
            }).addTo(samLayer);

            // Show area info
            const areaPx = data.area_pixels;
            const score = data.score;
            showStats(`
                <div style="font-size:13px">
                    <b>SAM Segment</b><br>
                    Area: ${areaPx.toLocaleString()} pixels<br>
                    Confidence: ${(score * 100).toFixed(1)}%<br>
                    Points: ${samPoints.length} (${samLabels.filter(l=>l===1).length} fg, ${samLabels.filter(l=>l===0).length} bg)
                </div>
            `);
        }
    } catch (err) {
        console.error("SAM predict error:", err);
    }
});

// ============================
// JOB SYSTEM
// ============================

async function runJob(endpoint, body) {
    showStatus(`Starting ${endpoint}...`);

    try {
        const resp = await fetch(`/api/${endpoint}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            let errMsg = "Unknown error";
            try {
                const err = await resp.json();
                errMsg = err.error || err.detail || errMsg;
            } catch {
                errMsg = await resp.text();
            }
            hideStatus();
            alert("Error: " + errMsg);
            return;
        }

        const { job_id } = await resp.json();
        pollJob(job_id, endpoint);
    } catch (err) {
        hideStatus();
        alert("Request failed: " + err.message);
    }
}

async function pollJob(jobId, endpoint) {
    const poll = async () => {
        try {
            const resp = await fetch(`/api/status/${jobId}`);
            const data = await resp.json();

            showStatus(data.message || data.status);

            if (data.status === "complete") {
                hideStatus();
                const result = await fetch(`/api/result/${jobId}`);
                const resultData = await result.json();
                displayResult(resultData, endpoint);
            } else if (data.status === "error") {
                hideStatus();
                alert("Processing failed: " + (data.error || "Unknown error"));
            } else {
                setTimeout(poll, 1500);
            }
        } catch (err) {
            hideStatus();
            console.error("Poll error:", err);
        }
    };
    poll();
}

function showStatus(msg) {
    const el = document.getElementById("job-status");
    document.getElementById("status-text").textContent = msg;
    el.classList.remove("hidden");
}

function hideStatus() {
    document.getElementById("job-status").classList.add("hidden");
}

// ============================
// DISPLAY RESULTS
// ============================

function displayResult(data, endpoint) {
    if (endpoint === "satellite-preview") {
        previewLayer.clearLayers();
        if (data.overlay_image) {
            const bounds = [[data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]];
            L.imageOverlay("data:image/png;base64," + data.overlay_image, bounds).addTo(previewLayer);
        }
        showStats('<div style="font-size:13px"><b>Satellite Preview</b><br>Image loaded. Size: ' +
            (data.shape ? data.shape[1] + "x" + data.shape[0] + " px" : "N/A") + "</div>");
    }

    else if (endpoint === "ndvi") {
        ndviLayer.clearLayers();
        if (data.overlay_image) {
            const bounds = [[data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]];
            L.imageOverlay("data:image/png;base64," + data.overlay_image, bounds, {
                opacity: 0.75,
            }).addTo(ndviLayer);
        }
        if (data.stats) {
            displayNdviStats(data.stats, data.indices);
        }
    }

    else if (endpoint === "cropmap") {
        cropLayer.clearLayers();
        const bounds = [[data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]];
        if (data.class_image) {
            L.imageOverlay("data:image/png;base64," + data.class_image, bounds, { opacity: 0.8 }).addTo(cropLayer);
            map.fitBounds(bounds);
        }
        // Legend
        let html = "<b>Crop classification</b>";
        html += '<div class="crop-legend">';
        for (const c of data.legend) {
            html += `<div class="legend-row"><span class="swatch" style="background:${c.color}"></span>` +
                    `<span class="legend-label">${c.label}</span><span class="legend-pct">${c.percent}%</span></div>`;
        }
        html += "</div>";
        showStats(html);

        setupSeasonSlider(data.monthly, bounds);
    }

    else if (endpoint === "timeseries") {
        let html = "<b>NDVI Time Series</b>";
        html += '<div id="ndvi-plot" style="width:100%;height:200px;margin-top:6px"></div>';
        if (data.phenology) {
            const ph = data.phenology;
            if (ph.peak_date) {
                html += `<div class="muted small" style="margin-top:4px">Peak ${ph.peak_ndvi} on ${ph.peak_date} &middot; amplitude ${ph.amplitude}</div>`;
            }
            html += `<div class="small" style="margin-top:4px">${ph.hint}</div>`;
        }
        showStats(html);
        renderNdviPlot(data.series || []);
    }

    else if (endpoint === "sam/load") {
        // SAM area loaded — enable click mode
        previewLayer.clearLayers();
        if (data.overlay_image) {
            const bounds = [[data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]];
            L.imageOverlay("data:image/png;base64," + data.overlay_image, bounds).addTo(previewLayer);
        }

        samMode = true;
        samPoints = [];
        samLabels = [];
        samPointsLayer.clearLayers();
        samLayer.clearLayers();
        document.getElementById("sam-controls").classList.remove("hidden");

        showStats('<div style="font-size:13px"><b>SAM Ready</b><br>Click on the map to segment objects.<br>Green = include, Red = exclude</div>');
    }
}

function displayNdviStats(stats, indices) {
    const colorForValue = (v) => {
        if (v > 0.5) return "#3fb950";
        if (v > 0.2) return "#d4a056";
        return "#f85149";
    };

    let html = "<b>Vegetation Indices</b>";
    html += '<div class="ndvi-stats">';
    html += `<div class="ndvi-stat"><div class="value" style="color:${colorForValue(stats.min)}">${stats.min.toFixed(3)}</div><div class="label">Min</div></div>`;
    html += `<div class="ndvi-stat"><div class="value" style="color:${colorForValue(stats.mean)}">${stats.mean.toFixed(3)}</div><div class="label">Mean</div></div>`;
    html += `<div class="ndvi-stat"><div class="value" style="color:${colorForValue(stats.max)}">${stats.max.toFixed(3)}</div><div class="label">Max</div></div>`;
    html += "</div>";

    if (indices) {
        html += '<div style="margin-top:8px;font-size:12px">';
        for (const [name, idx] of Object.entries(indices)) {
            html += `<div style="color:#8b949e">${name.toUpperCase()}: min=${idx.min.toFixed(3)}, mean=${idx.mean.toFixed(3)}, max=${idx.max.toFixed(3)}</div>`;
        }
        html += "</div>";
    }

    showStats(html);
}

function showStats(html) {
    document.getElementById("stats-display").innerHTML = html;
}

// Interactive NDVI time-series chart (Plotly): mean line + 25-75% band, hover values.
function renderNdviPlot(series) {
    if (!window.Plotly || !series.length) return;
    const x = series.map(p => p.date);
    const mean = series.map(p => p.ndvi_mean);
    const p25 = series.map(p => p.ndvi_p25 ?? p.ndvi_mean);
    const p75 = series.map(p => p.ndvi_p75 ?? p.ndvi_mean);

    const band = {
        x: x.concat([...x].reverse()),
        y: p75.concat([...p25].reverse()),
        fill: "toself", fillcolor: "rgba(22,163,74,0.15)",
        line: { color: "transparent" }, hoverinfo: "skip",
        showlegend: false, name: "25–75%",
    };
    const line = {
        x, y: mean, mode: "lines+markers", name: "mean NDVI",
        line: { color: "#16a34a", width: 2 }, marker: { size: 6 },
        hovertemplate: "%{x}<br>NDVI %{y:.3f}<extra></extra>",
    };
    const layout = {
        margin: { l: 34, r: 8, t: 6, b: 28 },
        height: 200,
        yaxis: { range: [-0.1, 1.0], gridcolor: "#eee", zeroline: false, tickfont: { size: 9 } },
        xaxis: { gridcolor: "#eee", tickfont: { size: 9 } },
        font: { size: 10, color: "#18181b" },
        paper_bgcolor: "white", plot_bgcolor: "white", showlegend: false,
        hoverlabel: { bgcolor: "#18181b", font: { color: "#fff", size: 11 } },
    };
    Plotly.newPlot("ndvi-plot", [band, line], layout,
        { responsive: true, displaylogo: false, displayModeBar: false });
}

// ============================
// SEASON SLIDER
// ============================

let seasonFrames = [];   // [{month, layer}]

function setupSeasonSlider(monthly, bounds) {
    seasonLayer.clearLayers();
    seasonFrames = (monthly || []).map(m => ({
        month: m.month,
        layer: L.imageOverlay("data:image/png;base64," + m.image, bounds, { opacity: 0.85 }),
    }));

    const panel = document.getElementById("season-slider");
    const range = document.getElementById("season-range");
    const label = document.getElementById("slider-month");

    if (seasonFrames.length === 0) { panel.classList.add("hidden"); return; }

    range.min = 0;
    range.max = seasonFrames.length - 1;
    range.value = seasonFrames.length - 1;  // start at latest month
    panel.classList.remove("hidden");

    const showFrame = (i) => {
        seasonLayer.clearLayers();
        seasonFrames[i].layer.addTo(seasonLayer);
        label.textContent = seasonFrames[i].month;
    };
    // Default view is the crop classification (the headline result); the
    // season NDVI overlay only appears once the user scrubs the slider.
    label.textContent = seasonFrames[seasonFrames.length - 1].month + " — drag to view";
    range.oninput = () => showFrame(parseInt(range.value, 10));
}

// ============================
// CHAT
// ============================

const chatInput = document.getElementById("chat-input");
const chatSend = document.getElementById("chat-send");
const chatMessages = document.getElementById("chat-messages");

// Check which chat provider is active
fetch("/api/chat/provider")
    .then(r => r.json())
    .then(data => {
        const badge = document.getElementById("chat-provider-badge");
        if (data.configured) {
            let label = data.provider;
            if (data.provider === "local" && data.local_model) label = data.local_model;
            badge.textContent = label;
            badge.style.background = "#238636";
            badge.style.color = "#fff";
        } else {
            badge.textContent = "not configured";
        }
    })
    .catch(() => {});

chatSend.addEventListener("click", sendChat);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
});

async function sendChat() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendChat("user", msg);
    chatInput.value = "";
    chatSend.disabled = true;

    // Show a "thinking" bubble so the wait doesn't look frozen, then replace
    // its text when the reply arrives.
    const thinking = appendChat("assistant", "•••");
    thinking.classList.add("thinking");

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: msg,
                bbox: currentBbox,
                session_id: "default",
            }),
        });
        const data = await resp.json();
        thinking.classList.remove("thinking");
        thinking.textContent = data.response;

        // The backend resolves navigation commands ("go to X") and bare place
        // names, returning a bbox — move the map there if so.
        if (data.bbox) setBbox(data.bbox);
    } catch (err) {
        thinking.classList.remove("thinking");
        thinking.textContent = "Connection error: " + err.message;
    }

    chatSend.disabled = false;
    chatInput.focus();
}

function appendChat(role, text) {
    const div = document.createElement("div");
    div.className = "chat-msg " + role;
    div.textContent = text;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return div;
}
