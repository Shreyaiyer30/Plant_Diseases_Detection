document.addEventListener('DOMContentLoaded', function() {
    const chatbotWidget = document.getElementById('chatbot-widget');
    const openChatBtn = document.getElementById('open-chatbot');
    const closeChatBtn = document.getElementById('close-chat');
    const sendMessageBtn = document.getElementById('send-message');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const chatTitle = document.getElementById('chat-disease-title');

    let chatHistory = [];

    openChatBtn.addEventListener('click', () => {
        chatbotWidget.style.display = 'flex';
        openChatBtn.style.display = 'none';
        scrollToBottom();
    });

    closeChatBtn.addEventListener('click', () => {
        chatbotWidget.style.display = 'none';
        openChatBtn.style.display = 'block';
    });

    sendMessageBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;

        // Get detected disease from hidden input or global variable
        const diseaseElement = document.getElementById('detected-disease');
        const disease = diseaseElement ? diseaseElement.value : 'Unknown';

        displayMessage(message, 'user');
        userInput.value = '';

        fetch('/api/chat', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question: message,
                disease: disease,
                chat_history: chatHistory
            })
        })
        .then(response => response.json())
        .then(data => {
            displayMessage(data.response, 'bot');
            if (data.meta?.disease && chatTitle) {
                chatTitle.textContent = data.meta.disease;
            }
            chatHistory.push({role: 'user', content: message});
            chatHistory.push({role: 'bot', content: data.response});
            
            // Save chat history to DB with source
            saveHistory(message, data.response, disease, data.source);
        })
        .catch(error => {
            console.error('Chat error:', error);
            displayMessage('Sorry, there was an error processing your request.', 'bot');
        });
    }

    function displayMessage(text, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = sender === 'user' ? 'user-message' : 'bot-message';
        messageDiv.textContent = text;
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function saveHistory(msg, resp, disease, source) {
        const diseaseElement = document.getElementById('detected-disease');
        const confidence = diseaseElement ? diseaseElement.dataset.confidence : 0;

        fetch('/api/chat/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question: msg,
                response: resp,
                disease: disease,
                source: source,
                confidence: confidence
            })
        }).catch(err => console.error('Save history error:', err));
    }

    // Quick action handling
    window.askQuickAction = function(action) {
        let question = "";
        switch(action) {
            case 'treatment': question = "Show treatment steps"; break;
            case 'prevention': question = "Prevention tips"; break;
            case 'organic': question = "Organic remedies"; break;
            case 'chemical': question = "Chemical options"; break;
        }
        if (question) {
            userInput.value = question;
            sendMessage();
        }
    };
});
