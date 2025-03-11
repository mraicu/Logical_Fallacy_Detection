(function () {

    async function sendTextToServer(text) {
        try {
            const classification = await new Promise((resolve) => {
                chrome.storage.sync.get("classification", (data) => {
                    resolve(data.classification || "binary"); // Default to "binary"
                });
            });

            console.log(classification)

            const response = await fetch("http://localhost:8110/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({text, classification}),
            });

            if (!response.ok) throw new Error("Server error");

            const data = await response.json();
            return data;
        } catch (error) {
            console.error("Error sending text to server:", error);
            return null;
        }
    }

    function highlightText(phrase, label, color) {
        if (!phrase) return;

        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
        const textNodes = [];

        while (walker.nextNode()) {
            if (walker.currentNode.nodeValue.trim().length > 0) {
                textNodes.push(walker.currentNode);
            }
        }

        let fullText = "", nodeMap = [];

        textNodes.forEach((node) => {
            nodeMap.push({node, start: fullText.length, end: fullText.length + node.nodeValue.length});
            fullText += node.nodeValue;
        });

        const matchIndex = fullText.toLowerCase().indexOf(phrase.toLowerCase());
        if (matchIndex === -1) return;

        const matchEnd = matchIndex + phrase.length, nodesToUpdate = [];

        nodeMap.forEach(({node, start, end}) => {
            if (matchIndex < end && matchEnd > start) {
                nodesToUpdate.push({
                    node,
                    startOffset: Math.max(0, matchIndex - start),
                    endOffset: Math.min(node.nodeValue.length, matchEnd - start)
                });
            }
        });

        nodesToUpdate.forEach(({node, startOffset, endOffset}) => {
            const text = node.nodeValue, parent = node.parentNode;
            const before = document.createTextNode(text.substring(0, startOffset));

            // Create a div for the label
            const labelDiv = document.createElement("div");
            labelDiv.textContent = label;
            Object.assign(labelDiv.style, {
                display: "inline",
                backgroundColor: color,
                color: color !== "none" ? "darkgrey" : "",
                fontWeight: "none",
                padding: "2px 4px",
                borderRadius: "4px 0px 0px 4px",
                // fontSize : parseFloat(window.getComputedStyle(node.parentNode).fontSize) - 5 +"px"
            });

            // Create a div for the highlighted text
            const highlightDiv = document.createElement("div");
            highlightDiv.textContent = text.substring(startOffset, endOffset);
            Object.assign(highlightDiv.style, {
                display: "inline",
                backgroundColor: color,
                color: color !== "none" ? "black" : "",
                padding: "2px 4px",
                borderRadius: "0px 4px 4px 0px"
            });

            const after = document.createTextNode(text.substring(endOffset));

            const fragment = document.createDocumentFragment();
            if (before.nodeValue) fragment.appendChild(before);
            fragment.appendChild(labelDiv);  // Add the label before the highlighted text
            fragment.appendChild(highlightDiv);
            if (after.nodeValue) fragment.appendChild(after);

            parent.replaceChild(fragment, node);
        });
    }


    // function highlightText(phrase, color) {
    //     if (!phrase) return;
    //
    //     const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    //     const textNodes = [];
    //
    //     while (walker.nextNode()) {
    //         if (walker.currentNode.nodeValue.trim().length > 0) {
    //             textNodes.push(walker.currentNode);
    //         }
    //     }
    //
    //     let fullText = "", nodeMap = [];
    //
    //     textNodes.forEach((node) => {
    //         nodeMap.push({node, start: fullText.length, end: fullText.length + node.nodeValue.length});
    //         fullText += node.nodeValue;
    //     });
    //
    //     const matchIndex = fullText.toLowerCase().indexOf(phrase.toLowerCase());
    //     if (matchIndex === -1) return;
    //
    //     const matchEnd = matchIndex + phrase.length, nodesToUpdate = [];
    //
    //     nodeMap.forEach(({node, start, end}) => {
    //         if (matchIndex < end && matchEnd > start) {
    //             nodesToUpdate.push({
    //                 node,
    //                 startOffset: Math.max(0, matchIndex - start),
    //                 endOffset: Math.min(node.nodeValue.length, matchEnd - start)
    //             });
    //         }
    //     });
    //
    //     nodesToUpdate.forEach(({node, startOffset, endOffset}) => {
    //         const text = node.nodeValue, parent = node.parentNode;
    //         const before = document.createTextNode(text.substring(0, startOffset));
    //         const highlightSpan = document.createElement("span");
    //         highlightSpan.textContent = text.substring(startOffset, endOffset);
    //         Object.assign(highlightSpan.style, {
    //             backgroundColor: color,
    //             color: color !== "none" ? "black" : ""
    //         });
    //         const after = document.createTextNode(text.substring(endOffset));
    //
    //         const fragment = document.createDocumentFragment();
    //         if (before.nodeValue) fragment.appendChild(before);
    //         fragment.appendChild(highlightSpan);
    //         if (after.nodeValue) fragment.appendChild(after);
    //
    //         parent.replaceChild(fragment, node);
    //     });
    // }

    function createSelectionButton() {
        const button = document.createElement("button");
        Object.assign(button.style, {
            position: "absolute",
            top: "0px",
            left: "0px",
            padding: "5px 10px",
            fontSize: "14px",
            backgroundColor: "#007BFF",
            color: "white",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer",
            boxShadow: "0px 2px 5px rgba(0, 0, 0, 0.3)",
            zIndex: "9999",
            opacity: "0",
            transition: "opacity 0.2s ease-in-out"
        });
        button.id = "selection-button";
        button.innerText = "Analyze";
        // button.addEventListener("click", () => alert("Button clicked!"));
        button.addEventListener("click", async () => {
            const selection = window.getSelection();
            if (!selection || selection.toString().trim() === "") return;

            const text = selection.toString();

            // Send selection to server
            const predictions = await sendTextToServer(text);

            console.log(predictions)

            if (predictions) {
                Object.entries(predictions.predictions).forEach(([sentence, [label, color]]) => {
                    // console.log(sentence, label, color)
                    highlightText(sentence, label, color);
                });
            }

            // TODO


            button.style.opacity = "0"; // Hide the button after highlighting

            // Clear text selection
            if (selection.removeAllRanges) {
                selection.removeAllRanges();
            } else if (selection.empty) {
                selection.empty();
            }
        });
        document.body.appendChild(button);
    }

    function updateSelectionButtonPosition() {
        const selection = window.getSelection();
        if (!selection || selection.toString().trim() === "") return;

        const range = selection.getRangeAt(0), rect = range.getBoundingClientRect();
        const button = document.getElementById("selection-button");
        if (button) {
            button.style.top = `${window.scrollY + rect.top - 40}px`;
            button.style.left = `${window.scrollX + rect.left}px`;
            button.style.opacity = "1";
        }
    }

    // Remove button when clicking elsewhere
    function hideSelectionButton(event) {
        setTimeout(() => {
            const button = document.getElementById("selection-button");
            const selection = window.getSelection();
            if (button && !button.contains(event.target) && selection.toString().trim() === "") {
                button.style.opacity = "0";
            }
        }, 100);
    }

    function setupEventListeners() {
        document.addEventListener("mouseup", updateSelectionButtonPosition);
        document.addEventListener("click", hideSelectionButton);
        // chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        //     if (request.action === "highlight") {
        //         highlightText(request.phrase);
        //         sendResponse({ success: true });
        //     }
        // });
    }

    function init() {
        createSelectionButton();
        setupEventListeners();
    }

    init();
})();



