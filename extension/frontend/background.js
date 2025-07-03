const API_URL = "http://localhost:8111";

chrome.storage.local.get("classification", (data) => {
    if (!data || !data.classification) {
        chrome.storage.local.set({classification: 'binary', sentiment: false});
    }
});

chrome.runtime.onMessage.addListener((request, _, sendResponse) => {
    switch (request.type) {
        case "ANALYZE_SELECTION_REQUEST":
            try {
                POST("/analyze", request.payload)
                    .then((response) => {
                        sendResponse({
                            type: "ANALYZE_SELECTION_RESPONSE",
                            payload: response,
                        });
                    })
                    .catch((error) => {
                        sendResponse({type: "ANALYZE_SELECTION_ERROR", payload: null});
                    });

                return true;
            } catch (error) {
                console.error("Analyze selection request error:", error);

                sendResponse({type: "ANALYZE_SELECTION_ERROR", payload: null});
            }
            break;
    }
});

/**
 * Send a POST request to an endpoint
 *
 * @param {string} endpoint - The endpoint to send the request to
 * @param {Object} payload - The request payload
 *
 * @returns {Object} - Response data
 * @throws Error
 */
async function POST(endpoint, payload) {
    try {
        const response = await fetch(API_URL + endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });
        if (!response.ok) throw new Error("Server error");

        const data = await response.json();

        return data;
    } catch (error) {
        throw new Error(error.message);
    }
}