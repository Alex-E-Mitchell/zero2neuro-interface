"use strict";

const earlyStopping = document.getElementById("early_stopping");
const earlyStoppingFields = document.getElementById("early-stopping-fields");

function updateEarlyStopping() {
    // Keep controls enabled so toggling off retains entered values on POST.
    // The server ignores these settings when early stopping is off.
    earlyStoppingFields.hidden = !earlyStopping.checked;
}

earlyStopping.addEventListener("change", updateEarlyStopping);
updateEarlyStopping();

const errors = document.getElementById("form-errors");
const output = document.getElementById("generated-config");
if (errors) errors.focus();
else if (output) output.focus();

const copyButton = document.getElementById("copy-button");
if (copyButton) {
    copyButton.addEventListener("click", async () => {
        const status = document.getElementById("copy-status");
        try {
            await navigator.clipboard.writeText(document.getElementById("config-text").textContent);
            status.textContent = "Configuration copied.";
        } catch {
            status.textContent = "Copy was unavailable. Select the configuration text and copy it manually.";
        }
    });
}
