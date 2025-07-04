/**
 * Create the Analyze button and add it to the page
 */
function createAnalyzeButton() {
    const websiteTheme = detectTheme();

    // Root container
    const root = document.createElement("div");
    root.id = "analyzeBtn";
    root.classList.add(
        "mr-analyze-btn-container",
        websiteTheme === "dark" ? "mr-analyze-btn-dark" : "mr-analyze-btn-light"
    );

    // Classifier button wrapper
    const classifierBtnWrapper = document.createElement("div");
    classifierBtnWrapper.classList.add("mr-classifier-btn-wrapper");

    // Classifier button
    const classifierBtn = document.createElement("button");
    classifierBtn.id = "classifierBtn";
    classifierBtn.innerText = "B";
    classifierBtn.classList.add("mr-classifier-btn");

    // expand button
    classifierBtn.addEventListener("mouseenter", () => {
        classifierBtnWrapper.style.width = "182px";
    });
    classifierBtnWrapper.addEventListener("mouseleave", () => {
        classifierBtnWrapper.style.width = "27px";
    });

    // Classifier select
    const classifierSelect = document.createElement("select");
    classifierSelect.classList.add("mr-classifier-select");

    classifierSelect.insertAdjacentHTML(
        "afterbegin",
        `
      <option value="binary">Binary: 2</option>
      <option value="3-classes">Multi: 3</option>
      <option value="5-classes">Multi: 5</option>
      <option value="15-classes">Multi: 15</option>
    `
    );

    classifierSelect.addEventListener("change", onClassifierChange);

    chrome.storage.local.get("classification", (data) => {
        if (data) {
            classifierSelect.value = data.classification;
            classifierBtn.innerText = getClassifierValueSymbol(data.classification);
        }
    });

    // Classifier sentiment checkbox
    const classifierSentimentDiv = document.createElement('div');
    classifierSentimentDiv.classList.add("mr-classifier-sentiment");

    const classifierSentimentCheckbox = document.createElement("input");
    classifierSentimentCheckbox.type = 'checkbox';
    classifierSentimentCheckbox.id = 'sentiment-checkbox';

    classifierSentimentCheckbox.addEventListener("change", onClassifierCheckboxChange);

    chrome.storage.local.get("sentiment", (data) => {
        if (data) {
            classifierSentimentCheckbox.checked = data.sentiment;
        }
    });

    // Classifier sentiment label
    const classifierSentimentLabel = document.createElement('label');
    classifierSentimentLabel.innerText = 'Sentiment';
    classifierSentimentLabel.htmlFor = 'sentiment-checkbox';

    // Analyze button
    const analyzeBtn = document.createElement("button");
    analyzeBtn.classList.add("mr-analyze-btn");
    analyzeBtn.addEventListener("click", onAnalyzeButtonClick);

    // Analyze button span
    const analyzeBtnSpan = document.createElement("span");
    analyzeBtnSpan.id = "analyzeBtnSpan";
    analyzeBtnSpan.innerText = "Analyze";
    analyzeBtnSpan.classList.add(
        "mr-analyze-btn-span",
        websiteTheme === "dark" ? "mr-text-light" : "mr-text-dark"
    );

    // Analyze button icon
    const analyzeBtnIcon = document.createElement("div");
    analyzeBtnIcon.id = "analyzeBtnIcon";
    analyzeBtnIcon.innerHTML = SVG_ARROW_RIGHT_ICON;
    analyzeBtnIcon.classList.add(
        "mr-analyze-btn-icon",
        websiteTheme === "dark" ? "mr-text-light" : "mr-text-dark"
    );

    // Append elements to their parents
    classifierSentimentDiv.appendChild(classifierSentimentCheckbox);
    classifierSentimentDiv.appendChild(classifierSentimentLabel);

    classifierBtnWrapper.appendChild(classifierBtn);
    classifierBtnWrapper.appendChild(classifierSelect);
    classifierBtnWrapper.appendChild(classifierSentimentDiv);

    analyzeBtn.appendChild(analyzeBtnSpan);
    analyzeBtn.appendChild(analyzeBtnIcon);

    root.appendChild(classifierBtnWrapper);
    root.appendChild(analyzeBtn);

    document.body.appendChild(root);
}

/**
 * When the Classifier select changes
 */
function onClassifierChange(event) {
    const selectedValue = event.target.value;
    if (!selectedValue) return;

    const classifierBtn = document.getElementById("classifierBtn");
    if (!classifierBtn) return;

    // Save option in chrome storage
    chrome.storage.local.set({classification: selectedValue});

    classifierBtn.innerText = getClassifierValueSymbol(selectedValue);
}

/**
 * When the Classifier checkbox changes
 */
function onClassifierCheckboxChange(event) {
    event.preventDefault();
    event.stopPropagation();
    const isChecked = event.target.checked;

    // Save option in chrome storage
    chrome.storage.local.set({sentiment: isChecked});
}

/**
 * When the Analyze button is clicked
 */
async function onAnalyzeButtonClick() {
    const [selectedText, selection] = getSelectedText();
    if (!selectedText) return;

    const analyzeBtn = document.getElementById("analyzeBtn");
    const analyzeBtnSpan = document.getElementById("analyzeBtnSpan");
    const analyzeBtnIcon = document.getElementById("analyzeBtnIcon");

    // Set loading state
    analyzeBtn.disabled = true;
    analyzeBtnSpan.innerText = "Loading";
    analyzeBtnSpan.style.opacity = 0.65;
    analyzeBtnIcon.innerHTML = SVG_LOADER_ICON;
    analyzeBtnIcon.classList.add("mr-loading-spinner");
    analyzeBtnIcon.style.opacity = 0.65;

    chrome.storage.local.get("classification", (classificationData) => {
        if (!classificationData || !classificationData.classification) return;

        chrome.storage.local.get("sentiment", (sentimentData) => {
            if (!sentimentData) return;

            chrome.runtime.sendMessage(
                {
                    type: "ANALYZE_SELECTION_REQUEST",
                    payload: {
                        selectedText,
                        classification: classificationData.classification,
                        sentiment: sentimentData.sentiment
                    },
                },
                (response) => {
                    if (!response || !response.payload) return;

                    // Reset analyze button after response
                    analyzeBtn.disabled = false;
                    analyzeBtnSpan.innerText = "Analyze";
                    analyzeBtnSpan.style.opacity = 1;
                    analyzeBtnIcon.innerHTML = SVG_ARROW_RIGHT_ICON;
                    analyzeBtnIcon.classList.remove("mr-loading-spinner");
                    analyzeBtnIcon.style.opacity = 1;

                    // Highlight sentences
                    const {predictions} = response.payload;
                    Object.entries(predictions).forEach(([sentence, [label, props]]) => {
                        highlightText(sentence, label, props);
                    });

                    // Hide button and clear selection
                    analyzeBtn.style.opacity = "0";
                    if (selection.removeAllRanges) {
                        selection.removeAllRanges();
                    } else if (selection.empty) {
                        selection.empty();
                    }
                }
            );
        });
    });
}
