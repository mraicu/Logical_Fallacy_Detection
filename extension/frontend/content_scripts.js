/**
 * Re-position the Analyze button on each text selection
 */
function updateAnalyzeButtonPosition() {
    const [selectedText, selection] = getSelectedText();
    if (!selectedText) return;

    const analyzeBtn = document.getElementById("analyzeBtn");
    if (!analyzeBtn) return;

    // Get selection bounding rectangle
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    // Update Analyze button position and opacity
    analyzeBtn.style.top = `${window.scrollY + rect.top - 45}px`;
    analyzeBtn.style.left = `${window.scrollX + rect.left}px`;

    analyzeBtn.style.opacity = "1";
}

/**
 * Hide button if clicked somewhere else in the page
 */
function hideAnalyzeButton(event) {
    setTimeout(() => {
        const analyzeBtn = document.getElementById("analyzeBtn");
        const selection = window.getSelection();
        if (
            analyzeBtn &&
            !analyzeBtn.contains(event.target) &&
            selection.toString().trim() === ""
        ) {
            analyzeBtn.style.opacity = "0";
        }
    }, 100);
}

/**
 * Init function
 */
(() => {
    // Create Analyze Button
    createAnalyzeButton();

    // Event Listeners
    document.addEventListener("mouseup", updateAnalyzeButtonPosition);
    document.addEventListener("click", hideAnalyzeButton);
})();
