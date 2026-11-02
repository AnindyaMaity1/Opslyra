(() => {
  const PNG_SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const CSS = `
    .accuracy-gridline { stroke: #e9edf2; stroke-width: 1; }
    .accuracy-axis-label, .accuracy-bar-label { fill: #687386; font: 10px Arial, sans-serif; }
    .accuracy-target { fill: #303c52; font: 600 12px Arial, sans-serif; }
    .accuracy-value { fill: #354158; font: 600 10px Arial, sans-serif; }
    .accuracy-bar-model { fill: #6371e9; }
    .accuracy-bar-baseline { fill: #9da6b6; }
    .conflict-bar-exact { fill: #6371e9; }
    .conflict-bar-near { fill: #e69b31; }
    .reliability-perfect-line { stroke: #98a2b3; stroke-width: 1.5; stroke-dasharray: 5 5; }
    .reliability-series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
    .reliability-series.reliability-uncalibrated { fill: none; stroke: #6975dc; }
    .reliability-series.reliability-sigmoid { fill: none; stroke: #28a28a; }
    .reliability-series.reliability-isotonic { fill: none; stroke: #e49a32; }
    .reliability-point.reliability-uncalibrated { fill: #6975dc; }
    .reliability-point.reliability-sigmoid { fill: #28a28a; }
    .reliability-point.reliability-isotonic { fill: #e49a32; }
    .reliability-point { stroke: #ffffff; stroke-width: 1.5; }
    .reliability-axis-title { fill: #687386; font: 10px Arial, sans-serif; }
  `;

  function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit += 1) {
        crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
      }
    }
    return (crc ^ 0xffffffff) >>> 0;
  }

  function chunk(type, data) {
    const typeBytes = new TextEncoder().encode(type);
    const output = new Uint8Array(12 + data.length);
    const view = new DataView(output.buffer);
    view.setUint32(0, data.length);
    output.set(typeBytes, 4);
    output.set(data, 8);
    const crcInput = new Uint8Array(typeBytes.length + data.length);
    crcInput.set(typeBytes);
    crcInput.set(data, typeBytes.length);
    view.setUint32(8 + data.length, crc32(crcInput));
    return output;
  }

  function addDpiMetadata(pngBytes, dpi) {
    for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
      if (pngBytes[index] !== PNG_SIGNATURE[index]) {
        throw new Error("The browser returned an invalid PNG image.");
      }
    }
    const pixelsPerMeter = Math.round(dpi / 0.0254);
    const physData = new Uint8Array(9);
    const physView = new DataView(physData.buffer);
    physView.setUint32(0, pixelsPerMeter);
    physView.setUint32(4, pixelsPerMeter);
    physData[8] = 1;
    const physChunk = chunk("pHYs", physData);
    const view = new DataView(pngBytes.buffer, pngBytes.byteOffset, pngBytes.byteLength);
    let offset = PNG_SIGNATURE.length;
    let insertionOffset = -1;
    const parts = [pngBytes.slice(0, PNG_SIGNATURE.length)];

    while (offset + 12 <= pngBytes.length) {
      const length = view.getUint32(offset);
      const end = offset + length + 12;
      if (end > pngBytes.length) {
        throw new Error("The browser returned a truncated PNG image.");
      }
      const type = new TextDecoder().decode(pngBytes.subarray(offset + 4, offset + 8));
      if (type === "pHYs") {
        parts.push(physChunk);
      } else {
        parts.push(pngBytes.slice(offset, end));
      }
      if (type === "IHDR") insertionOffset = parts.length;
      offset = end;
      if (type === "IEND") break;
    }
    if (offset !== pngBytes.length || insertionOffset < 0) {
      throw new Error("The browser returned a malformed PNG image.");
    }
    if (!parts.some((part) => {
      return part.length >= 8 && new TextDecoder().decode(part.subarray(4, 8)) === "pHYs";
    })) {
      parts.splice(insertionOffset, 0, physChunk);
    }
    const result = new Uint8Array(parts.reduce((size, part) => size + part.length, 0));
    let writeOffset = 0;
    for (const part of parts) {
      result.set(part, writeOffset);
      writeOffset += part.length;
    }
    return result;
  }

  function canvasPng(canvas) {
    return new Promise((resolve, reject) => {
      canvas.toBlob((blob) => {
        if (blob) resolve(blob);
        else reject(new Error("The browser could not encode the high-resolution PNG."));
      }, "image/png");
    });
  }

  function loadSvgImage(svgBlob) {
    const sourceUrl = URL.createObjectURL(svgBlob);
    const image = new Image();
    return new Promise((resolve, reject) => {
      image.onload = () => {
        URL.revokeObjectURL(sourceUrl);
        resolve(image);
      };
      image.onerror = () => {
        URL.revokeObjectURL(sourceUrl);
        reject(new Error("The browser could not render the graph SVG."));
      };
      image.src = sourceUrl;
    });
  }

  async function downloadGraph(button) {
    const chart = document.getElementById(button.dataset.chartId);
    if (!(chart instanceof SVGSVGElement)) {
      throw new Error("The graph could not be found.");
    }
    const dpi = Number(button.dataset.dpi);
    const widthInches = Number(button.dataset.widthInches);
    if (!Number.isFinite(dpi) || dpi < 1 || !Number.isFinite(widthInches) || widthInches <= 0) {
      throw new Error("The graph export settings are invalid.");
    }
    const viewBox = chart.viewBox.baseVal;
    if (!viewBox.width || !viewBox.height) {
      throw new Error("The graph has no valid viewBox for export.");
    }
    const width = Math.round(widthInches * dpi);
    const height = Math.round(width * viewBox.height / viewBox.width);
    const clone = chart.cloneNode(true);
    clone.setAttribute("width", String(viewBox.width));
    clone.setAttribute("height", String(viewBox.height));
    const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
    style.textContent = CSS;
    clone.insertBefore(style, clone.firstChild);
    const background = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    background.setAttribute("x", String(viewBox.x));
    background.setAttribute("y", String(viewBox.y));
    background.setAttribute("width", String(viewBox.width));
    background.setAttribute("height", String(viewBox.height));
    background.setAttribute("fill", "#ffffff");
    clone.insertBefore(background, style);

    const source = new Blob([new XMLSerializer().serializeToString(clone)], {
      type: "image/svg+xml;charset=utf-8",
    });
    const image = await loadSvgImage(source);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");
    if (!context) {
      image.src = "";
      throw new Error("The browser could not create an image surface for export.");
    }
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    image.src = "";
    const png = await canvasPng(canvas);
    canvas.width = 0;
    canvas.height = 0;
    const bytes = addDpiMetadata(new Uint8Array(await png.arrayBuffer()), dpi);
    const link = document.createElement("a");
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
    link.href = url;
    link.download = `${button.dataset.filename || "graph"}-${dpi}dpi.png`;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  document.querySelectorAll(".graph-download").forEach((button) => {
    button.addEventListener("click", async () => {
      const panel = button.closest(".graph-panel");
      const status = panel?.querySelector(".graph-download-status");
      button.disabled = true;
      if (status) status.textContent = "Rendering 1200-DPI image…";
      try {
        await downloadGraph(button);
        if (status) status.textContent = "PNG downloaded with 1200-DPI metadata.";
      } catch (error) {
        if (status) status.textContent = error instanceof Error ? error.message : "Graph export failed.";
      } finally {
        button.disabled = false;
      }
    });
  });
})();
