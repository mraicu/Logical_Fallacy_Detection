/**
 * Get the selection text
 *
 * @returns {Array<string, Selection>} [selectedText, selection] - An array with 2 elements: selectedText and selection
 */
function getSelectedText() {
    const selection = window.getSelection();
    if (!selection) return [];

    const selectedText = selection.toString().trim();
    if (!selectedText || selectedText === "") return [];

    return [selectedText, selection];
}

/**
 * Get classifier symbol by the classifier value
 *
 * @param {string} selectedValue - Classifier value
 *
 * @returns {string} - Symbol
 */
function getClassifierValueSymbol(selectedValue) {
    let symbol = "B";

    switch (selectedValue) {
        case "binary":
            symbol = "2";
            break;

        case "3-classes":
            symbol = "3";
            break;

        case "5-classes":
            symbol = "5";
            break;

        case "15-classes":
            symbol = "15";
            break;
    }

    return symbol;
}

/**
 * Detect dark or light theme
 */
function getLuminance(color) {
    let rgb = color.match(/\d+/g).map(Number);
    if (!rgb || rgb.length < 3) return 1;

    let [r, g, b] = rgb.map((v) => v / 255);

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function detectTheme() {
    let bodyStyles = window.getComputedStyle(document.body);
    let bgColor = bodyStyles.backgroundColor || "rgb(255,255,255)";
    let textColor = bodyStyles.color || "rgb(0,0,0)";

    let bgLuminance = getLuminance(bgColor);
    let textLuminance = getLuminance(textColor);

    return bgLuminance < textLuminance ? "dark" : "light";
}

const SVG_ARROW_RIGHT_ICON = `
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-feather mr-feather-arrow-right"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
`;

const SVG_LOADER_ICON = `
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-feather mr-feather-loader"><line x1="12" y1="2" x2="12" y2="6"></line><line x1="12" y1="18" x2="12" y2="22"></line><line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line><line x1="2" y1="12" x2="6" y2="12"></line><line x1="18" y1="12" x2="22" y2="12"></line><line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line></svg>
`;

const SVG_INFO_ICON = `
<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="mr-feather mr-feather-info"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
`;
