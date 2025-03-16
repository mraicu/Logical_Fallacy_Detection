function highlightText(phrase, label, color) {
  if (!phrase) return;

  const walker = document.createTreeWalker(
    document.body,
    NodeFilter.SHOW_TEXT,
    null,
    false
  );
  const textNodes = [];

  while (walker.nextNode()) {
    if (walker.currentNode.nodeValue.trim().length > 0) {
      textNodes.push(walker.currentNode);
    }
  }

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

  const matchIndex = fullText.toLowerCase().indexOf(phrase.toLowerCase());
  if (matchIndex === -1) return;

  const matchEnd = matchIndex + phrase.length,
    nodesToUpdate = [];

  nodeMap.forEach(({ node, start, end }) => {
    if (matchIndex < end && matchEnd > start) {
      nodesToUpdate.push({
        node,
        startOffset: Math.max(0, matchIndex - start),
        endOffset: Math.min(node.nodeValue.length, matchEnd - start),
      });
    }
  });

  nodesToUpdate.forEach(({ node, startOffset, endOffset }) => {
    const text = node.nodeValue,
      parent = node.parentNode;
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
      borderRadius: "0px 4px 4px 0px",
    });

    const after = document.createTextNode(text.substring(endOffset));

    const fragment = document.createDocumentFragment();
    if (before.nodeValue) fragment.appendChild(before);
    fragment.appendChild(labelDiv); // Add the label before the highlighted text
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
