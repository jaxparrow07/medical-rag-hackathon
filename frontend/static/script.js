const queryInput = document.getElementById('query-input');
const askButton = document.getElementById('ask-button');
const answerSection = document.getElementById('answer-section');
const answerContent = document.getElementById('answer-content');
const contextSection = document.getElementById('context-section');
const contextCards = document.getElementById('context-cards');

// Handle query submission
askButton.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleQuery();
    }
});

async function handleQuery() {
    const query = queryInput.value.trim();
    
    if (!query) {
        alert('Please enter a question');
        return;
    }
    
    // Disable input during processing
    askButton.disabled = true;
    askButton.querySelector('.btn-text').textContent = 'Thinking...';
    
    // Show answer section with shimmer
    answerSection.classList.remove('hidden');
    answerSection.querySelector('.answer-card').classList.add('thinking');
    answerContent.textContent = 'Processing your question...';
    
    // Hide context initially
    contextSection.classList.add('hidden');
    
    try {
        // Make API call
        const response = await fetch('/api/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query })
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove shimmer effect
        answerSection.querySelector('.answer-card').classList.remove('thinking');
        
        // Display answer with formatting
        const formattedAnswer = formatText(data.answer || 'No answer generated');
        answerContent.innerHTML = formattedAnswer;
        
        // Display context cards
        if (data.context && data.context.length > 0) {
            contextSection.classList.remove('hidden');
            contextCards.innerHTML = '';
            
            data.context.forEach((ctx, index) => {
                const card = document.createElement('div');
                card.className = 'context-card';
                
                // Extract text from context
                const contextText = ctx.text || ctx.content || ctx.page_content || 'No content';
                const similarity = ctx.similarity || ctx.score || null;
                
                card.innerHTML = `
                    <div class="context-source">Source ${index + 1}</div>
                    <div class="context-text">${escapeHtml(contextText)}</div>
                    ${similarity !== null ? `<div class="context-similarity">Similarity: ${(similarity * 100).toFixed(1)}%</div>` : ''}
                `;
                contextCards.appendChild(card);
            });
        }
        
    } catch (error) {
        console.error('Error:', error);
        answerSection.querySelector('.answer-card').classList.remove('thinking');
        answerContent.textContent = '❌ Error: ' + error.message;
    } finally {
        // Re-enable input
        askButton.disabled = false;
        askButton.querySelector('.btn-text').textContent = 'Ask';
    }
}

// Helper function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Helper function to format text with markdown-style notation
function formatText(text) {
    // Escape HTML first
    let formatted = escapeHtml(text);
    
    // Parse **bold** (double asterisks)
    formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    
    // Parse *italic* (single asterisks, but not already part of **)
    formatted = formatted.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');
    
    // Parse _underline_ (underscores)
    formatted = formatted.replace(/_(.+?)_/g, '<u>$1</u>');
    
    // Preserve line breaks
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}

// Auto-resize textarea
queryInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});
