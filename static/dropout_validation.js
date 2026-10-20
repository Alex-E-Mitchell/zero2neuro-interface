// HTML max is inclusive, and browsers may round it to "1" in native messages.
// Keep the numeric constraints, but explain the exclusive limit explicitly.
for (const [id, label] of [["dropout", "Dropout rate"], ["dropout_input", "Input dropout rate"]]) {
    const input = document.getElementById(id);
    function updateDropoutValidity() {
        const rate = input.valueAsNumber;
        const invalid = input.validity.badInput ||
            (input.value !== "" && (!Number.isFinite(rate) || rate < 0 || rate >= 1));
        input.setCustomValidity(invalid ? `${label} must be at least 0 and less than 1.` : "");
    }
    input.addEventListener("input", updateDropoutValidity);
    input.addEventListener("invalid", updateDropoutValidity);
    updateDropoutValidity();
}
