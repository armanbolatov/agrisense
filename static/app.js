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
const changeLayer = L.layerGroup().addTo(map);

const overlayLayers = {
    "Satellite Preview": previewLayer,
    "NDVI": ndviLayer,
    "Crop Map": cropLayer,
    "Season (NDVI)": seasonLayer,
    "Change": changeLayer,
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
    [previewLayer, ndviLayer, cropLayer, seasonLayer, changeLayer, samLayer, samPointsLayer]
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
// YEAR + TOOL BUTTONS
// ============================

const yearSelect = document.getElementById("year-select");
const changeYearA = document.getElementById("change-year-a");
const changeYearB = document.getElementById("change-year-b");
for (let y = 2024; y >= 2018; y--) {
    yearSelect.add(new Option(y, y, y === 2023, y === 2023));
    changeYearB.add(new Option(y, y, y === 2023, y === 2023));
    changeYearA.add(new Option(y, y, y === 2020, y === 2020));
}

function getYear() { return parseInt(yearSelect.value, 10); }

// Keep the season date inputs in sync with the chosen year (month/day fixed).
function syncDatesToYear() {
    const y = getYear();
    [["date-spring-start", "04-20"], ["date-spring-end", "05-31"],
     ["date-peak-start", "07-01"], ["date-peak-end", "07-31"],
     ["date-late-start", "09-01"], ["date-late-end", "09-30"]].forEach(([id, md]) => {
        document.getElementById(id).value = `${y}-${md}`;
    });
}
yearSelect.addEventListener("change", syncDatesToYear);

function getDateRanges() {
    return [
        document.getElementById("date-spring-start").value + "/" + document.getElementById("date-spring-end").value,
        document.getElementById("date-peak-start").value + "/" + document.getElementById("date-peak-end").value,
        document.getElementById("date-late-start").value + "/" + document.getElementById("date-late-end").value,
    ];
}

// Single entry point — the chat can call this too (agentic chat).
function runAnalysis(tool) {
    if (!currentBbox) { alert("Select an area on the map first (draw a rectangle or pick a region)."); return; }
    if (tool === "preview") runJob("satellite-preview", { bbox: currentBbox, date_ranges: getDateRanges() });
    else if (tool === "ndvi") runJob("ndvi", { bbox: currentBbox, date_ranges: getDateRanges() });
    else if (tool === "timeseries") runJob("timeseries", { bbox: currentBbox, year: getYear() });
    else if (tool === "cropmap") runJob("cropmap", { bbox: currentBbox, year: getYear() });
    else if (tool === "sam") runJob("sam/load", { bbox: currentBbox, date_range: getDateRanges()[1] });
    else if (tool === "change") runJob("change", {
        bbox: currentBbox,
        year_a: parseInt(changeYearA.value, 10),
        year_b: parseInt(changeYearB.value, 10),
    });
}

document.getElementById("btn-preview").addEventListener("click", () => runAnalysis("preview"));
document.getElementById("btn-ndvi").addEventListener("click", () => runAnalysis("ndvi"));
document.getElementById("btn-timeseries").addEventListener("click", () => runAnalysis("timeseries"));
document.getElementById("btn-cropmap").addEventListener("click", () => runAnalysis("cropmap"));
document.getElementById("btn-sam-load").addEventListener("click", () => runAnalysis("sam"));
document.getElementById("btn-change").addEventListener("click", () => runAnalysis("change"));

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
            lastSamGeojson = data.geojson;

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
                <div class="export-links"><a href="#" onclick="exportSamGeojson();return false">⬇ GeoJSON</a></div>
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
        html += geotiffLink("cropmap");
        showStats(html);

        setupSeasonSlider(data.monthly, bounds);
    }

    else if (endpoint === "change") {
        changeLayer.clearLayers();
        const bounds = [[data.bbox[1], data.bbox[0]], [data.bbox[3], data.bbox[2]]];
        if (data.overlay_image) {
            L.imageOverlay("data:image/png;base64," + data.overlay_image, bounds, { opacity: 0.8 }).addTo(changeLayer);
            map.fitBounds(bounds);
        }
        const s = data.stats || {};
        let html = `<b>Change ${data.year_a} → ${data.year_b}</b>`;
        html += '<div class="ndvi-stats">';
        html += `<div class="ndvi-stat"><div class="value" style="color:#16a34a">${s.greener_pct ?? 0}%</div><div class="label">greener</div></div>`;
        html += `<div class="ndvi-stat"><div class="value" style="color:#dc2626">${s.browner_pct ?? 0}%</div><div class="label">browner</div></div>`;
        html += `<div class="ndvi-stat"><div class="value">${s.mean_change ?? 0}</div><div class="label">Δ NDVI</div></div>`;
        html += "</div>";
        html += '<div class="small muted" style="margin-top:6px">Green = NDVI increase, red = decrease (peak season).</div>';
        html += geotiffLink("change");
        showStats(html);
    }

    else if (endpoint === "timeseries") {
        lastSeries = data.series || [];
        let html = "<b>NDVI Time Series</b>";
        html += '<div id="ndvi-plot" style="width:100%;height:210px;margin-top:6px"></div>';
        if (data.phenology) {
            const ph = data.phenology;
            if (ph.peak_date) {
                html += `<div class="muted small" style="margin-top:4px">Peak ${ph.peak_ndvi} on ${ph.peak_date} &middot; amplitude ${ph.amplitude}</div>`;
            }
            html += `<div class="small" style="margin-top:4px">${ph.hint}</div>`;
        }
        html += '<div class="export-links"><a href="#" onclick="exportSeriesCsv();return false">⬇ CSV</a></div>';
        showStats(html);
        renderNdviPlot(lastSeries);
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
    html += geotiffLink("ndvi");

    showStats(html);
}

function showStats(html) {
    document.getElementById("stats-display").innerHTML = html;
}

// ============================
// EXPORT
// ============================

let lastSeries = null;
let lastSamGeojson = null;

function downloadFile(filename, content, mime) {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

window.exportSeriesCsv = () => {
    if (!lastSeries) return;
    const cols = ["date", "cloud_cover", "ndvi_mean", "ndvi_median", "ndvi_p25",
                  "ndvi_p75", "ndvi_max", "ndwi_mean", "evi_mean"];
    const rows = [cols.join(",")];
    lastSeries.forEach(p => rows.push(cols.map(c => p[c] ?? "").join(",")));
    downloadFile("ndvi_timeseries.csv", rows.join("\n"), "text/csv");
};

window.exportSamGeojson = () => {
    if (lastSamGeojson) downloadFile("sam_segment.geojson", JSON.stringify(lastSamGeojson, null, 2), "application/geo+json");
};

function geotiffLink(kind) {
    return `<div class="export-links"><a href="/api/export/${kind}.tif">⬇ GeoTIFF</a></div>`;
}

// Interactive time-series chart (Plotly): NDVI (mean + 25–75% band) plus
// NDWI and EVI traces you can toggle via the legend.
function renderNdviPlot(series) {
    if (!window.Plotly || !series.length) return;
    const x = series.map(p => p.date);
    const ndvi = series.map(p => p.ndvi_mean);
    const p25 = series.map(p => p.ndvi_p25 ?? p.ndvi_mean);
    const p75 = series.map(p => p.ndvi_p75 ?? p.ndvi_mean);

    const traces = [
        {   // 25–75% NDVI band
            x: x.concat([...x].reverse()),
            y: p75.concat([...p25].reverse()),
            fill: "toself", fillcolor: "rgba(22,163,74,0.13)",
            line: { color: "transparent" }, hoverinfo: "skip", showlegend: false,
        },
        {
            x, y: ndvi, mode: "lines+markers", name: "NDVI",
            line: { color: "#16a34a", width: 2 }, marker: { size: 6 },
            hovertemplate: "%{x}<br>NDVI %{y:.3f}<extra></extra>",
        },
    ];
    if (series[0].ndwi_mean !== undefined) {
        traces.push({
            x, y: series.map(p => p.ndwi_mean), mode: "lines+markers", name: "NDWI",
            line: { color: "#1f6feb", width: 1.5 }, marker: { size: 4 },
            visible: "legendonly",
            hovertemplate: "%{x}<br>NDWI %{y:.3f}<extra></extra>",
        });
    }
    if (series[0].evi_mean !== undefined) {
        traces.push({
            x, y: series.map(p => p.evi_mean), mode: "lines+markers", name: "EVI",
            line: { color: "#e8590c", width: 1.5 }, marker: { size: 4 },
            visible: "legendonly",
            hovertemplate: "%{x}<br>EVI %{y:.3f}<extra></extra>",
        });
    }
    const layout = {
        margin: { l: 34, r: 8, t: 6, b: 20 },
        height: 210,
        yaxis: { range: [-0.4, 1.0], gridcolor: "#eee", zeroline: false, tickfont: { size: 9 } },
        xaxis: { gridcolor: "#eee", tickfont: { size: 9 } },
        font: { size: 10, color: "#18181b" },
        paper_bgcolor: "white", plot_bgcolor: "white",
        showlegend: true,
        legend: { orientation: "h", y: -0.18, font: { size: 10 } },
        hoverlabel: { bgcolor: "#18181b", font: { color: "#fff", size: 11 } },
    };
    Plotly.newPlot("ndvi-plot", traces, layout,
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

// Apply a nav/action result: move the map and/or run an analysis tool.
function applyChatMeta(d) {
    if (d.bbox) setBbox(d.bbox);
    if (d.action) setTimeout(() => runAnalysis(d.action), 250);
}

async function sendChat() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendChat("user", msg);
    chatInput.value = "";
    chatSend.disabled = true;

    const bubble = appendChat("assistant", "•••");
    bubble.classList.add("thinking");
    const body = JSON.stringify({ message: msg, bbox: currentBbox, session_id: "default" });

    try {
        const resp = await fetch("/api/chat/stream", {
            method: "POST", headers: { "Content-Type": "application/json" }, body,
        });
        if (!resp.ok || !resp.body) throw new Error("stream unavailable");

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = "", text = "", streaming = false;

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const parts = buf.split("\n\n");
            buf = parts.pop();
            for (const part of parts) {
                const line = part.trim();
                if (!line.startsWith("data:")) continue;
                let ev;
                try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
                if (ev.type === "meta") {
                    bubble.classList.remove("thinking");
                    bubble.textContent = ev.response || "";
                    applyChatMeta(ev);
                } else if (ev.type === "token") {
                    if (!streaming) { bubble.classList.remove("thinking"); bubble.textContent = ""; streaming = true; }
                    text += ev.t;
                    bubble.textContent = text;
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            }
        }
        if (bubble.classList.contains("thinking")) { bubble.classList.remove("thinking"); bubble.textContent = text || "…"; }
    } catch (err) {
        // Fall back to the non-streaming endpoint.
        try {
            const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body });
            const data = await r.json();
            bubble.classList.remove("thinking");
            bubble.textContent = data.response;
            applyChatMeta(data);
        } catch (e2) {
            bubble.classList.remove("thinking");
            bubble.textContent = "Connection error: " + e2.message;
        }
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
