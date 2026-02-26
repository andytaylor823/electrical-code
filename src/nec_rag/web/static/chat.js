/* ===================================================================
   NEC Code Expert — Chat Frontend Logic
   =================================================================== */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

function generateSessionId() {
    return Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 10);
}
let sessionId = generateSessionId();
let pendingImages = [];   // { file: File, dataUrl: string }
let isWaiting = false;

// Streaming response state (active during token-by-token text delivery)
let streamingAccumulator = "";
let streamingBodyEl = null;
let streamingRow = null;
let renderScheduled = false;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------

const loginScreen   = document.getElementById("login-screen");
const chatScreen    = document.getElementById("chat-screen");
const loginForm     = document.getElementById("login-form");
const passwordInput = document.getElementById("password-input");
const loginBtn      = document.getElementById("login-btn");
const loginError    = document.getElementById("login-error");

const messagesEl    = document.getElementById("messages");
const messageInput  = document.getElementById("message-input");
const sendBtn       = document.getElementById("send-btn");
const imageInput    = document.getElementById("image-input");
const previewsEl    = document.getElementById("image-previews");
const newChatBtn    = document.getElementById("new-chat-btn");
const contextWheel  = document.getElementById("context-wheel");
const wheelFill     = document.getElementById("wheel-fill");
const wheelTooltip  = document.getElementById("wheel-tooltip");

const feedbackBtn       = document.getElementById("feedback-btn");
const feedbackOverlay   = document.getElementById("feedback-overlay");
const feedbackForm      = document.getElementById("feedback-form");
const feedbackInput     = document.getElementById("feedback-input");
const feedbackSubmitBtn = document.getElementById("feedback-submit-btn");
const feedbackCancelBtn = document.getElementById("feedback-cancel-btn");
const feedbackError     = document.getElementById("feedback-error");
const feedbackSuccess   = document.getElementById("feedback-success");

// ---------------------------------------------------------------------------
// Markdown setup
// ---------------------------------------------------------------------------

if (typeof marked !== "undefined") {
    marked.use({ breaks: true, gfm: true });
}

function renderMarkdown(text) {
    if (typeof marked !== "undefined") {
        try { return marked.parse(text); } catch (_e) { /* fall through */ }
    }
    // Minimal fallback: escape HTML and convert newlines
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\n/g, "<br>");
}

// ---------------------------------------------------------------------------
// Section reference detection and popover
// ---------------------------------------------------------------------------

// Client-side caches for section and table lookups
const _sectionCache = new Map();
const _tableCache = new Map();

function fetchSectionData(sectionId) {
    if (_sectionCache.has(sectionId)) return _sectionCache.get(sectionId);
    const promise = fetch(`/api/section/${encodeURIComponent(sectionId)}`)
        .then(resp => {
            if (!resp.ok) throw new Error(`Section ${sectionId} not found`);
            return resp.json();
        })
        .catch(err => {
            _sectionCache.delete(sectionId);
            throw err;
        });
    _sectionCache.set(sectionId, promise);
    return promise;
}

function fetchTableData(tableId) {
    if (_tableCache.has(tableId)) return _tableCache.get(tableId);
    const promise = fetch(`/api/table/${encodeURIComponent(tableId)}`)
        .then(resp => {
            if (!resp.ok) throw new Error(`Table ${tableId} not found`);
            return resp.json();
        })
        .catch(err => {
            _tableCache.delete(tableId);
            throw err;
        });
    _tableCache.set(tableId, promise);
    return promise;
}

/**
 * Walk text nodes and wrap NEC section/table references in interactive spans.
 *
 * "Table 690.31(A)(3)(1)" -> <span class="nec-ref" data-ref-type="table" data-ref-id="690.31(A)(3)(1)">
 * "Section 705.12"        -> <span class="nec-ref" data-ref-type="section" data-ref-id="705.12">
 * "705.12"  (bare)        -> <span class="nec-ref" data-ref-type="section" data-ref-id="705.12">
 *
 * Table matches take priority so "Table 690.31" is never misidentified as a section.
 * Article numbers in the NEC start at 90, so requiring >= 90 avoids false
 * positives on measurements like "1.8 m" or "30.5 volts".
 */
function postProcessSectionRefs(container) {
    // Combined pattern with two alternatives (table first for priority):
    //   1) "Table " + digits.digits + optional (A)(3)(1)-style suffixes
    //   2) Optional "Section "/"§" + digits.digits + optional suffixes
    const NEC_REF_RE = /(?:(Table)\s+((?:9\d|[1-9]\d{2})\.\d+(?:\([A-Za-z0-9]+\))*))|(?:(?:Section|§)\s*)?((\b(?:9\d|[1-9]\d{2})\.\d+)(?:\([A-Za-z0-9]+\))*)/gi;

    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null);
    const textNodes = [];
    while (walker.nextNode()) {
        const parent = walker.currentNode.parentNode;
        if (parent.closest && (parent.closest(".nec-ref") || parent.closest("code") || parent.closest("pre"))) continue;
        if (NEC_REF_RE.test(walker.currentNode.nodeValue)) {
            textNodes.push(walker.currentNode);
        }
        NEC_REF_RE.lastIndex = 0;
    }

    for (const node of textNodes) {
        const text = node.nodeValue;
        NEC_REF_RE.lastIndex = 0;
        const frag = document.createDocumentFragment();
        let lastIndex = 0;
        let match;

        while ((match = NEC_REF_RE.exec(text)) !== null) {
            if (match.index > lastIndex) {
                frag.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
            }
            const fullMatch = match[0];
            const isTable = !!match[1];
            // For tables: match[2] is the full ID with suffixes (e.g. "690.31(A)(3)(1)")
            // For sections: match[4] is the base digits.digits ID
            const refId = isTable ? match[2] : match[4];

            const span = document.createElement("span");
            span.className = "nec-ref";
            span.dataset.refType = isTable ? "table" : "section";
            span.dataset.refId = refId;
            span.textContent = fullMatch;
            frag.appendChild(span);

            lastIndex = match.index + fullMatch.length;
        }

        if (lastIndex < text.length) {
            frag.appendChild(document.createTextNode(text.slice(lastIndex)));
        }

        node.parentNode.replaceChild(frag, node);
    }
}

// ---------------------------------------------------------------------------
// Popover (single shared DOM element)
// ---------------------------------------------------------------------------

let _popoverEl = null;       // the popover DOM element
let _popoverPinned = false;  // true when the user clicked to persist
let _popoverTarget = null;   // the .nec-section-ref span that triggered it
let _hideTimeout = null;     // delay before hiding on mouseleave

function _ensurePopover() {
    if (_popoverEl) return _popoverEl;
    _popoverEl = document.createElement("div");
    _popoverEl.className = "nec-popover hidden";
    _popoverEl.innerHTML = `
        <div class="nec-popover-header"></div>
        <div class="nec-popover-body"></div>
    `;
    // Keep the popover open while the cursor is over it (hover mode)
    _popoverEl.addEventListener("mouseenter", () => { clearTimeout(_hideTimeout); });
    _popoverEl.addEventListener("mouseleave", () => { if (!_popoverPinned) _scheduleHide(); });
    document.body.appendChild(_popoverEl);
    return _popoverEl;
}

function _scheduleHide() {
    clearTimeout(_hideTimeout);
    _hideTimeout = setTimeout(() => { _dismissPopover(); }, 200);
}

function _dismissPopover() {
    clearTimeout(_hideTimeout);
    if (!_popoverEl) return;
    _popoverEl.classList.add("hidden");
    _popoverEl.classList.remove("pinned", "table-mode");
    _popoverPinned = false;
    _popoverTarget = null;
}

function _positionPopover(refEl) {
    const pop = _ensurePopover();
    const rect = refEl.getBoundingClientRect();
    // Tables need more room; sections are narrower
    const isTable = pop.classList.contains("table-mode");
    const popWidth = isTable ? 520 : 380;

    // Horizontal: centre on the ref, clamped to viewport
    let left = rect.left + rect.width / 2 - popWidth / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - popWidth - 8));

    // Vertical: prefer above, fall below if not enough room
    pop.style.width = popWidth + "px";
    pop.style.left = left + "px";

    // Temporarily show off-screen to measure height
    pop.style.top = "-9999px";
    pop.classList.remove("hidden");
    const popHeight = pop.offsetHeight;

    const spaceAbove = rect.top;
    const spaceBelow = window.innerHeight - rect.bottom;
    if (spaceAbove >= popHeight + 8 || spaceAbove >= spaceBelow) {
        pop.style.top = (rect.top + window.scrollY - popHeight - 6) + "px";
    } else {
        pop.style.top = (rect.bottom + window.scrollY + 6) + "px";
    }
}

function _showPopover(refEl, pinned) {
    clearTimeout(_hideTimeout);
    const pop = _ensurePopover();
    const header = pop.querySelector(".nec-popover-header");
    const body = pop.querySelector(".nec-popover-body");

    const refType = refEl.dataset.refType;
    const refId = refEl.dataset.refId;

    _popoverTarget = refEl;
    _popoverPinned = pinned;

    // Show loading state
    header.textContent = refType === "table" ? `Table ${refId}` : `Section ${refId}`;
    body.textContent = "Loading\u2026";
    pop.classList.toggle("pinned", pinned);
    pop.classList.toggle("table-mode", refType === "table");
    _positionPopover(refEl);

    if (refType === "table") {
        fetchTableData(refId).then(data => {
            if (_popoverTarget !== refEl) return;
            let meta = "";
            if (data.article_num) meta += `Article ${data.article_num}`;
            if (data.page) meta += `${meta ? " \u00b7 " : ""}p.\u2009${data.page}`;
            const metaHtml = meta ? `<span class="nec-popover-meta">${meta}</span>` : "";
            header.innerHTML = `<strong>${_escHtml(data.title)}</strong>${metaHtml}`;
            body.innerHTML = renderMarkdown(pinned ? data.markdown : _truncateText(data.markdown, 400));
            _positionPopover(refEl);
        }).catch(() => { _dismissPopover(); });
    } else {
        fetchSectionData(refId).then(data => {
            if (_popoverTarget !== refEl) return;
            header.innerHTML = `<strong>Section ${data.id}</strong><span class="nec-popover-meta">Article ${data.article_num} &middot; p.\u2009${data.page}</span>`;
            body.textContent = pinned ? data.text : _truncateText(data.text, 300);
            _positionPopover(refEl);
        }).catch(() => { _dismissPopover(); });
    }
}

function _truncateText(text, maxLen) {
    return text.length > maxLen ? text.slice(0, maxLen) + "\u2026" : text;
}

function _escHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Event delegation on the messages container (hover + click on section refs)
// ---------------------------------------------------------------------------

messagesEl.addEventListener("mouseenter", (e) => {
    const ref = e.target.closest(".nec-ref");
    if (!ref || _popoverPinned) return;
    _showPopover(ref, false);
}, true);

messagesEl.addEventListener("mouseleave", (e) => {
    const ref = e.target.closest(".nec-ref");
    if (!ref || _popoverPinned) return;
    _scheduleHide();
}, true);

messagesEl.addEventListener("click", (e) => {
    const ref = e.target.closest(".nec-ref");
    if (!ref) return;
    e.preventDefault();
    if (_popoverPinned && _popoverTarget === ref) {
        _dismissPopover();
        return;
    }
    _showPopover(ref, true);
});

document.addEventListener("click", (e) => {
    if (!_popoverPinned) return;
    if (_popoverEl && _popoverEl.contains(e.target)) return;
    if (e.target.closest(".nec-ref")) return;
    _dismissPopover();
});

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && _popoverPinned) _dismissPopover();
});

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

async function handleLogin() {
    const password = passwordInput.value.trim();
    if (!password) return;

    loginBtn.disabled = true;
    loginError.textContent = "";

    try {
        const resp = await fetch("/api/login", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ password }),
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            loginError.textContent = data.detail || "Invalid password";
            return;
        }
        // Successful login — switch to chat screen
        loginScreen.classList.add("hidden");
        chatScreen.classList.remove("hidden");
        messageInput.focus();
    } catch (err) {
        loginError.textContent = "Connection error. Is the server running?";
    } finally {
        loginBtn.disabled = false;
    }
}

loginBtn.addEventListener("click", (e) => {
    e.preventDefault();
    handleLogin();
});

passwordInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        handleLogin();
    }
});

// ---------------------------------------------------------------------------
// Auto-resize textarea
// ---------------------------------------------------------------------------

messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + "px";
    updateSendButton();
});

function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    const hasImages = pendingImages.length > 0;
    sendBtn.disabled = (!hasText && !hasImages) || isWaiting;
}

// ---------------------------------------------------------------------------
// Image attachment
// ---------------------------------------------------------------------------

imageInput.addEventListener("change", () => {
    for (const file of imageInput.files) {
        if (pendingImages.length >= 5) break;

        const reader = new FileReader();
        reader.onload = (e) => {
            pendingImages.push({ file, dataUrl: e.target.result });
            renderPreviews();
            updateSendButton();
        };
        reader.readAsDataURL(file);
    }
    // Clear the input so the same file can be re-selected
    imageInput.value = "";
});

function renderPreviews() {
    previewsEl.innerHTML = "";
    pendingImages.forEach((img, i) => {
        const item = document.createElement("div");
        item.className = "image-preview-item";
        item.innerHTML = `
            <img src="${img.dataUrl}" alt="Preview">
            <button class="image-preview-remove" data-index="${i}">&times;</button>
        `;
        previewsEl.appendChild(item);
    });
}

previewsEl.addEventListener("click", (e) => {
    const removeBtn = e.target.closest(".image-preview-remove");
    if (!removeBtn) return;
    const index = parseInt(removeBtn.dataset.index, 10);
    pendingImages.splice(index, 1);
    renderPreviews();
    updateSendButton();
});

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------

sendBtn.addEventListener("click", sendMessage);
messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!sendBtn.disabled) sendMessage();
    }
});

async function sendMessage() {
    const text = messageInput.value.trim();
    const images = [...pendingImages];

    if (!text && images.length === 0) return;
    if (isWaiting) return;

    // Clear input immediately
    messageInput.value = "";
    messageInput.style.height = "auto";
    pendingImages = [];
    renderPreviews();

    // Hide the welcome message on first send
    const welcome = messagesEl.querySelector(".welcome-message");
    if (welcome) welcome.remove();

    // Render user message
    addMessageToUI("user", text, images.map(img => img.dataUrl));

    // Show the status indicator (replaces the old bouncing-dots thinking indicator)
    const statusRow = addStatusIndicator();
    const statusContainer = statusRow.querySelector(".thinking-status");
    isWaiting = true;
    updateSendButton();

    // Build form data
    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("message", text);
    for (const img of images) {
        formData.append("images", img.file);
    }

    try {
        const resp = await fetch("/api/chat", {
            method: "POST",
            body: formData,
        });

        if (resp.status === 401) {
            statusRow.remove();
            chatScreen.classList.add("hidden");
            loginScreen.classList.remove("hidden");
            loginError.textContent = "Session expired. Please log in again.";
            passwordInput.focus();
            return;
        }

        if (!resp.ok) {
            statusRow.remove();
            const data = await resp.json().catch(() => ({}));
            addMessageToUI("assistant", `**Error:** ${data.detail || "Something went wrong. Please try again."}`);
            return;
        }

        // Read the SSE stream and update the status indicator as events arrive
        await readSSEStream(resp, statusRow, statusContainer);
    } catch (err) {
        statusRow.remove();
        addMessageToUI("assistant", "**Error:** Could not reach the server. Is it still running?");
    } finally {
        isWaiting = false;
        updateSendButton();
        messageInput.focus();
    }
}

// ---------------------------------------------------------------------------
// SSE stream reader
// ---------------------------------------------------------------------------

async function readSSEStream(resp, statusRow, statusContainer) {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE events are separated by double newlines
        const parts = buffer.split("\n\n");
        buffer = parts.pop();

        for (const part of parts) {
            const trimmed = part.trim();
            if (!trimmed.startsWith("data: ")) continue;
            try {
                const data = JSON.parse(trimmed.slice(6));
                handleStreamEvent(data, statusRow, statusContainer);
            } catch (_e) {
                // Skip malformed SSE lines
            }
        }
    }

    // Process any remaining buffered data
    if (buffer.trim().startsWith("data: ")) {
        try {
            const data = JSON.parse(buffer.trim().slice(6));
            handleStreamEvent(data, statusRow, statusContainer);
        } catch (_e) {
            // Ignore
        }
    }
}

// ---------------------------------------------------------------------------
// Streaming response helpers
// ---------------------------------------------------------------------------

function beginStreamingResponse(statusRow) {
    // Create the assistant message row that tokens will stream into
    streamingRow = document.createElement("div");
    streamingRow.className = "message-row assistant";
    streamingRow.innerHTML = `
        <div class="message-avatar">\u26A1</div>
        <div class="message-content">
            <div class="message-role">NEC Expert</div>
            <div class="message-body streaming"></div>
        </div>
    `;
    statusRow.remove();
    messagesEl.appendChild(streamingRow);
    streamingBodyEl = streamingRow.querySelector(".message-body");
    streamingAccumulator = "";
    renderScheduled = false;
}

function scheduleStreamingRender() {
    if (renderScheduled) return;
    renderScheduled = true;
    requestAnimationFrame(() => {
        if (streamingBodyEl) {
            streamingBodyEl.innerHTML = renderMarkdown(streamingAccumulator);
            scrollToBottom();
        }
        renderScheduled = false;
    });
}

function finalizeStreamingResponse(tokenInfo, fullResponse) {
    // Use the server's authoritative full text if provided, otherwise keep the accumulator
    const finalText = fullResponse || streamingAccumulator;
    if (streamingBodyEl) {
        streamingBodyEl.innerHTML = renderMarkdown(finalText);
        streamingBodyEl.classList.remove("streaming");
        postProcessSectionRefs(streamingBodyEl);
    }
    if (tokenInfo) {
        updateContextWheel(tokenInfo);
    }
    scrollToBottom();

    // Reset streaming state
    streamingAccumulator = "";
    streamingBodyEl = null;
    streamingRow = null;
    renderScheduled = false;
}

// ---------------------------------------------------------------------------
// SSE event handler — updates cumulative shadow text
// ---------------------------------------------------------------------------

function handleStreamEvent(data, statusRow, statusContainer) {
    if (data.type === "thinking") {
        // Only add "Thinking…" if the last item isn't already a thinking indicator
        const lastItem = statusContainer.lastElementChild;
        if (lastItem && lastItem.classList.contains("active") && lastItem.textContent === "Thinking\u2026") {
            return;
        }
        const item = document.createElement("div");
        item.className = "status-item active";
        item.textContent = "Thinking\u2026";
        statusContainer.appendChild(item);
        scrollToBottom();

    } else if (data.type === "tool_start") {
        // Replace the active "Thinking…" with the tool description (present tense)
        const lastItem = statusContainer.lastElementChild;
        if (lastItem && lastItem.classList.contains("active")) {
            lastItem.textContent = data.description;
        } else {
            const item = document.createElement("div");
            item.className = "status-item active";
            item.textContent = data.description;
            statusContainer.appendChild(item);
        }
        scrollToBottom();

    } else if (data.type === "tool_end") {
        // Update the active item to past tense and mark completed
        const activeItem = statusContainer.querySelector(".status-item.active");
        if (activeItem) {
            activeItem.textContent = data.description;
            activeItem.classList.remove("active");
            activeItem.classList.add("completed");
        }
        scrollToBottom();

    } else if (data.type === "text_delta") {
        // First delta: transition from status indicator to streaming message bubble
        if (!streamingBodyEl) {
            beginStreamingResponse(statusRow);
        }
        streamingAccumulator += data.content;
        scheduleStreamingRender();

    } else if (data.type === "final") {
        if (streamingBodyEl) {
            // We were streaming — finalize with authoritative text and token info
            finalizeStreamingResponse(data.token_info, data.response);
        } else {
            // No deltas received (e.g. empty response) — fall back to full render
            statusRow.remove();
            addMessageToUI("assistant", data.response, [], data.token_info);
        }

    } else if (data.type === "error") {
        // Clean up streaming state if active
        if (streamingBodyEl) {
            streamingBodyEl.classList.remove("streaming");
            streamingAccumulator = "";
            streamingBodyEl = null;
            streamingRow = null;
        }
        statusRow.remove();
        addMessageToUI("assistant", `**Error:** ${data.detail || "Something went wrong. Please try again."}`);
    }
}

// ---------------------------------------------------------------------------
// Render messages
// ---------------------------------------------------------------------------

function addMessageToUI(role, text, imageDataUrls = [], tokenInfo = null) {
    const row = document.createElement("div");
    row.className = `message-row ${role}`;

    const avatarLabel = role === "user" ? "You" : "⚡";
    const roleName = role === "user" ? "You" : "NEC Expert";

    // Build image thumbnails HTML
    let imagesHtml = "";
    if (imageDataUrls.length > 0) {
        const imgs = imageDataUrls.map(url => `<img src="${url}" alt="Attached">`).join("");
        imagesHtml = `<div class="message-images">${imgs}</div>`;
    }

    // Render assistant messages as markdown, user messages as plain text
    let bodyHtml;
    if (role === "assistant") {
        bodyHtml = renderMarkdown(text);
    } else {
        bodyHtml = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/\n/g, "<br>");
    }

    row.innerHTML = `
        <div class="message-avatar">${avatarLabel}</div>
        <div class="message-content">
            <div class="message-role">${roleName}</div>
            ${imagesHtml}
            <div class="message-body">${bodyHtml}</div>
        </div>
    `;

    // Update the global context wheel after assistant responses
    if (tokenInfo) {
        updateContextWheel(tokenInfo);
    }

    messagesEl.appendChild(row);

    // Make section references interactive in assistant messages
    if (role === "assistant") {
        postProcessSectionRefs(row.querySelector(".message-body"));
    }

    scrollToBottom();
}

function addStatusIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
        <div class="message-avatar">⚡</div>
        <div class="message-content">
            <div class="message-role">NEC Expert</div>
            <div class="thinking-status"></div>
        </div>
    `;
    messagesEl.appendChild(row);
    scrollToBottom();
    return row;
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ---------------------------------------------------------------------------
// Context usage wheel
// ---------------------------------------------------------------------------

// SVG circle circumference = 2 * PI * r where r = 15.9155 ≈ 100
const WHEEL_CIRCUMFERENCE = 100;

function updateContextWheel(tokenInfo) {
    if (!tokenInfo || !tokenInfo.context_window) return;

    const used = tokenInfo.context_used || 0;
    const total = tokenInfo.context_window;
    const pct = Math.min(used / total, 1);

    // Set the fill amount via stroke-dashoffset (100 = empty, 0 = full)
    const offset = WHEEL_CIRCUMFERENCE * (1 - pct);
    wheelFill.setAttribute("stroke-dashoffset", offset.toFixed(1));

    // Color tiers: muted (<50%), accent (50-80%), warning orange (>80%)
    let color;
    if (pct < 0.5) {
        color = "var(--text-muted)";
    } else if (pct < 0.8) {
        color = "var(--accent)";
    } else {
        color = "#e57c1a";
    }
    wheelFill.style.stroke = color;

    const pctDisplay = Math.round(pct * 100);
    wheelTooltip.textContent = `${used.toLocaleString()} / ${total.toLocaleString()} tokens (${pctDisplay}%)`;

    // Show the wheel (hidden until first response)
    contextWheel.classList.remove("hidden");
}

function resetContextWheel() {
    contextWheel.classList.add("hidden");
    wheelFill.setAttribute("stroke-dashoffset", WHEEL_CIRCUMFERENCE);
    wheelFill.style.stroke = "";
    wheelTooltip.textContent = "";
}

// ---------------------------------------------------------------------------
// New Chat
// ---------------------------------------------------------------------------

newChatBtn.addEventListener("click", async () => {
    if (isWaiting) return;

    try {
        await fetch("/api/new-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId }),
        });
    } catch {
        // Ignore — we reset locally regardless
    }

    // Reset local state
    sessionId = generateSessionId();
    pendingImages = [];
    renderPreviews();
    resetContextWheel();

    // Clear messages and show welcome
    messagesEl.innerHTML = `
        <div class="welcome-message">
            <div class="welcome-icon">⚡</div>
            <h2>NEC Code Expert</h2>
            <p>Ask me anything about the NFPA 70 National Electrical Code, 2023 Edition.
               I can search specific sections, look up tables, and analyze images of
               electrical installations.</p>
        </div>
    `;

    messageInput.value = "";
    messageInput.style.height = "auto";
    updateSendButton();
    messageInput.focus();
});

// ---------------------------------------------------------------------------
// Feedback overlay
// ---------------------------------------------------------------------------

function openFeedback() {
    feedbackInput.value = "";
    feedbackError.textContent = "";
    feedbackSubmitBtn.disabled = true;
    feedbackOverlay.classList.remove("hidden");
    feedbackInput.focus();
}

function closeFeedback() {
    feedbackOverlay.classList.add("hidden");
    // Reset to form view (in case success message was showing)
    feedbackForm.classList.remove("hidden");
    feedbackSuccess.classList.add("hidden");
}

feedbackBtn.addEventListener("click", openFeedback);
feedbackCancelBtn.addEventListener("click", closeFeedback);

// Roadmap "Submit feedback!" link opens the same overlay
document.addEventListener("click", (e) => {
    const link = e.target.closest(".roadmap-feedback-link");
    if (!link) return;
    e.preventDefault();
    openFeedback();
});

// Close overlay when clicking the backdrop (not the card itself)
feedbackOverlay.addEventListener("click", (e) => {
    if (e.target === feedbackOverlay) closeFeedback();
});

// Enable/disable submit based on textarea content
feedbackInput.addEventListener("input", () => {
    feedbackSubmitBtn.disabled = feedbackInput.value.trim().length === 0;
});

feedbackSubmitBtn.addEventListener("click", async () => {
    const text = feedbackInput.value.trim();
    if (!text) return;

    feedbackSubmitBtn.disabled = true;
    feedbackError.textContent = "";

    try {
        const resp = await fetch("/api/feedback", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, feedback_text: text }),
        });

        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            feedbackError.textContent = data.detail || "Something went wrong.";
            feedbackSubmitBtn.disabled = false;
            return;
        }

        // Swap form for success message, then auto-close after a short delay
        feedbackForm.classList.add("hidden");
        feedbackSuccess.classList.remove("hidden");
        setTimeout(() => {
            closeFeedback();
        }, 1500);
    } catch {
        feedbackError.textContent = "Could not reach the server.";
        feedbackSubmitBtn.disabled = false;
    }
});

// ---------------------------------------------------------------------------
// Check auth on page load (skip login if cookie is still valid)
// ---------------------------------------------------------------------------

(async function checkAuth() {
    try {
        const resp = await fetch("/api/new-chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: "__ping__" }),
        });
        if (resp.ok) {
            // Already authenticated
            loginScreen.classList.add("hidden");
            chatScreen.classList.remove("hidden");
            messageInput.focus();
        }
    } catch {
        // Server not reachable or not authenticated — stay on login
    }
})();
