const themeToggle = document.getElementById('theme-toggle');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatMessages = document.getElementById('chat-messages');
const imageUpload = document.getElementById('image-upload');
const uploadBtn = document.getElementById('upload-btn');
const uploadPreview = document.getElementById('upload-preview');
const previewFilename = document.getElementById('preview-filename');
const clearUploadBtn = document.getElementById('clear-upload');

let sessionId = crypto.randomUUID();

// Theme setup
const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
if (prefersDark) {
    document.documentElement.setAttribute('data-theme', 'dark');
    themeToggle.innerHTML = '<i class="ri-sun-line"></i>';
}

themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    if (currentTheme === 'dark') {
        document.documentElement.removeAttribute('data-theme');
        themeToggle.innerHTML = '<i class="ri-moon-line"></i>';
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
        themeToggle.innerHTML = '<i class="ri-sun-line"></i>';
    }
});

// Image Upload handling
uploadBtn.addEventListener('click', () => {
    imageUpload.click();
});

imageUpload.addEventListener('change', () => {
    if (imageUpload.files.length > 0) {
        previewFilename.textContent = imageUpload.files[0].name;
        uploadPreview.classList.remove('hidden');
    }
});

clearUploadBtn.addEventListener('click', () => {
    imageUpload.value = '';
    uploadPreview.classList.add('hidden');
});

function scrollToBottom() {
    const chatContainer = document.querySelector('.chat-container');
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Parses <think> blocks into an accordion
function formatContent(content) {
    let formatted = content;
    
    // Replace <think>...</think> with an accordion
    const thinkRegex = /<think>([\s\S]*?)<\/think>/gi;
    formatted = formatted.replace(thinkRegex, (match, p1) => {
        return `
            <details class="think-block">
                <summary>Thinking Process...</summary>
                <div class="think-content">${p1.trim()}</div>
            </details>
        `;
    });
    
    return marked.parse(formatted);
}

function addMessage(role, content, imageUrl = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    let avatarIcon = role === 'assistant' ? 'ri-robot-2-line' : 'ri-user-3-line';
    
    let contentHtml = '';
    if (imageUrl) {
        contentHtml += `<img src="${imageUrl}" class="uploaded-image" alt="Uploaded product">`;
    }
    
    if (content) {
        contentHtml += formatContent(content);
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="${avatarIcon}"></i>
        </div>
        <div class="message-content">
            ${contentHtml}
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

function showSkeleton() {
    const skeletonDiv = document.createElement('div');
    skeletonDiv.className = 'message assistant skeleton-message';
    skeletonDiv.innerHTML = `
        <div class="message-avatar">
            <i class="ri-robot-2-line"></i>
        </div>
        <div class="message-content">
            <div class="skeleton-loader">
                <div class="skeleton-line"></div>
                <div class="skeleton-line"></div>
                <div class="skeleton-line short"></div>
            </div>
        </div>
    `;
    chatMessages.appendChild(skeletonDiv);
    scrollToBottom();
    return skeletonDiv;
}

function removeSkeleton(skeletonElement) {
    if (skeletonElement && skeletonElement.parentNode) {
        skeletonElement.parentNode.removeChild(skeletonElement);
    }
}

chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const text = chatInput.value.trim();
    const hasImage = imageUpload.files.length > 0;
    
    if (!text && !hasImage) return;
    
    // UI Updates
    chatInput.value = '';
    
    if (hasImage) {
        const file = imageUpload.files[0];
        const imageUrl = URL.createObjectURL(file);
        addMessage('user', text ? `*Uploaded image: ${file.name}*\n\n${text}` : `*Uploaded image: ${file.name}*`, imageUrl);
        
        // Hide preview
        imageUpload.value = '';
        uploadPreview.classList.add('hidden');
        
        const skeleton = showSkeleton();
        
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);
            
            const response = await fetch('/api/upload_image', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            removeSkeleton(skeleton);
            addMessage('assistant', data.response);
        } catch (error) {
            console.error('Error:', error);
            removeSkeleton(skeleton);
            addMessage('assistant', "Sorry, I encountered an error while processing your image.");
        }
    } else {
        addMessage('user', text);
        const skeleton = showSkeleton();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: text }],
                    session_id: sessionId
                })
            });
            
            const data = await response.json();
            removeSkeleton(skeleton);
            addMessage('assistant', data.response);
        } catch (error) {
            console.error('Error:', error);
            removeSkeleton(skeleton);
            addMessage('assistant', "Sorry, I encountered an error while processing your request.");
        }
    }
});
