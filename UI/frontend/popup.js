document.addEventListener("DOMContentLoaded", function () {
    const select = document.getElementById("classification-select");
    const saveBtn = document.getElementById("save-btn");
    const status = document.getElementById("status");

    // Load saved selection from Chrome storage
    chrome.storage.sync.get("classification", function (data) {
        if (data.classification) {
            select.value = data.classification;
        }
    });

    // Save the selected classification type
    saveBtn.addEventListener("click", function () {
        const selectedValue = select.value;
        chrome.storage.sync.set({ classification: selectedValue }, function () {
            status.textContent = "Saved!";
            setTimeout(() => (status.textContent = ""), 2000);
             window.close();
        });
    });
});
