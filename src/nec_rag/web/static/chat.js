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

    } else if (data.type === "final") {
        // Remove the status indicator and render the final response
        statusRow.remove();
        addMessageToUI("assistant", data.response, [], data.token_info);

    } else if (data.type === "error") {
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

    // Token usage badge for assistant messages
    let tokenHtml = "";
    if (tokenInfo && tokenInfo.total_tokens) {
        tokenHtml = `<div class="token-badge">${tokenInfo.total_tokens.toLocaleString()} tokens</div>`;
    }

    row.innerHTML = `
        <div class="message-avatar">${avatarLabel}</div>
        <div class="message-content">
            <div class="message-role">${roleName}</div>
            ${imagesHtml}
            <div class="message-body">${bodyHtml}</div>
            ${tokenHtml}
        </div>
    `;

    messagesEl.appendChild(row);
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
