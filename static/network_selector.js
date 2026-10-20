const networkType = document.getElementById("network_type");
const fullyConnectedFields = document.getElementById("fully-connected-fields");
const cnnFields = document.getElementById("cnn-fields");
const cnnStatus = document.getElementById("cnn-status");
const generateButton = document.getElementById("generate-button");

function updateNetworkType() {
    const isFullyConnected = networkType.value === "fully_connected";
    const isCnn = networkType.value === "cnn";
    fullyConnectedFields.hidden = !isFullyConnected;
    fullyConnectedFields.disabled = !isFullyConnected;
    // These draft fields have no browser validation; keep them enabled so FC POSTs retain their values.
    cnnFields.hidden = !isCnn;
    cnnStatus.hidden = !isCnn;
    generateButton.disabled = !isFullyConnected;
    document.getElementById("input_shape").required = isFullyConnected;
    document.getElementById("output_shape").required = isFullyConnected;
    document.getElementById("fc-input-help").hidden = !isFullyConnected;
    document.getElementById("cnn-input-help").hidden = !isCnn;
    document.getElementById("input_shape").placeholder = isCnn ? "e.g. 128, 128, 3" : "e.g. 34";
}

networkType.addEventListener("change", () => {
    updateNetworkType();
    // Results and errors belong to the previously submitted network type.
    for (const id of ["generated-config", "form-errors"]) {
    const element = document.getElementById(id);
        if (element) element.hidden = true;
    }
});

updateNetworkType();
