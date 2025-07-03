function highlightText(phrase, label, props) {
    if (!phrase) return;

    const websiteTheme = detectTheme();

    const color = props[0];
    const info = props[1];

    // Uses a TreeWalker to traverse all text nodes in the body of the document
    const walker = document.createTreeWalker(
        document.body,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );
    const textNodes = [];

    // Collect all non-empty text nodes
    while (walker.nextNode()) {
        if (walker.currentNode.nodeValue.trim().length > 0) {
            textNodes.push(walker.currentNode);
        }
    }

    // Concatenate all text node contents into one string
    // and map each node to its start/end position in the full text
    let fullText = "",
        nodeMap = [];

    textNodes.forEach((node) => {
        nodeMap.push({
            node,
            start: fullText.length,
            end: fullText.length + node.nodeValue.length,
        });
        fullText += node.nodeValue;
    });

    // Case-insensitive search for the phrase
    const matchIndex = fullText.toLowerCase().indexOf(phrase.toLowerCase());
    if (matchIndex === -1) return;

    const matchEnd = matchIndex + phrase.length,
        nodesToUpdate = [];

    // Calculates the exact start and end positions within each node
    nodeMap.forEach(({node, start, end}) => {
        if (matchIndex < end && matchEnd > start) {
            nodesToUpdate.push({
                node,
                startOffset: Math.max(0, matchIndex - start),
                endOffset: Math.min(node.nodeValue.length, matchEnd - start),
            });
        }
    });

    // For each affected node, replace part of it with highlighted elements
    nodesToUpdate.forEach(({node, startOffset, endOffset}) => {
        const text = node.nodeValue,
            parent = node.parentNode;

        // Split the original text into: before match, matched text, after match
        const before = document.createTextNode(text.substring(0, startOffset));

        // Create a div for the label
        const labelDiv = document.createElement("div");
        labelDiv.textContent = label;
        Object.assign(labelDiv.style, {
            position: "relative",
            display: "inline",
            backgroundColor: color,
            color: color !== "none" ? "darkgrey" : "",
            fontWeight: "none",
            padding: "2px 4px",
            borderRadius: "4px 0px 0px 4px",
            // fontSize : parseFloat(window.getComputedStyle(node.parentNode).fontSize) - 5 +"px"
        });

        // Create the info box
        const infoDiv = document.createElement('div');
        Object.assign(infoDiv.style, {
            position: "absolute",
            top: "32px",
            left: "0px",
            display: "none",
            backgroundColor: websiteTheme === "dark" ? "white" : "#444",
            padding: "6px 8px",
            borderRadius: "4px",
            zIndex: 9,
            flexDirection: "column",
            rowGap: "4px"
        });
        labelDiv.onmouseover = () => {
            infoDiv.style.display = "flex";
        };
        labelDiv.onmouseout = () => {
            infoDiv.style.display = "none";
        }

        // Info icon
        const labelInfoIcon = document.createElement("div");
        labelInfoIcon.innerHTML = SVG_INFO_ICON;
        labelInfoIcon.classList.add(
            "mr-label-info-icon",
            websiteTheme === "dark" ? "mr-text-darkgrey" : "mr-text-light"
        );

        // Info text
        const labelInfoText = document.createElement("div");
        labelInfoText.innerText = info;
        Object.assign(labelInfoText.style, {
            color: websiteTheme === "dark" ? "darkgrey" : "white",
            fontSize: "13px",
            lineHeight: 1.25,
        });

        infoDiv.appendChild(labelInfoIcon);
        infoDiv.appendChild(labelInfoText);

        // Add info div as child to labelDiv
        labelDiv.appendChild(infoDiv);

        // Create a div for the highlighted text
        const highlightDiv = document.createElement("div");
        highlightDiv.textContent = text.substring(startOffset, endOffset);
        Object.assign(highlightDiv.style, {
            display: "inline",
            backgroundColor: color,
            color: color !== "none" ? "black" : "",
            padding: "2px 4px",
            borderRadius: "0px 4px 4px 0px",
        });

        const after = document.createTextNode(text.substring(endOffset));

        // Replace the original text node with the composed fragment
        const fragment = document.createDocumentFragment();
        if (before.nodeValue) fragment.appendChild(before);
        fragment.appendChild(labelDiv); // Add the label before the highlighted text
        fragment.appendChild(highlightDiv);
        if (after.nodeValue) fragment.appendChild(after);

        parent.replaceChild(fragment, node);
    });
}