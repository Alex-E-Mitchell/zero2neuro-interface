// Run with: node --test python/tests/network_selector.test.js
// Execute the production script against a small DOM fixture; no browser packages needed.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../../static/network_selector.js"), "utf8");

function page(initialType, includeResult = true) {
    const elements = Object.fromEntries([
        "network_type", "fully-connected-fields", "cnn-fields", "cnn-status", "generate-button",
        "input_shape", "output_shape", "fc-input-help", "cnn-input-help",
        "conv_number_filters", "conv_kernel_size", "conv_pool_size", "conv_padding", "conv_activation",
    ].map(id => [id, { hidden: false, disabled: false, required: false }]));
    elements.network_type.value = initialType;
    elements.input_shape.value = "34";
    elements.output_shape.value = "10, 20";
    elements.conv_number_filters.value = "8, 16";
    elements.conv_kernel_size.value = "3, 3";
    elements.conv_pool_size.value = "0, 2";
    elements.conv_padding.value = "same";
    elements.conv_activation.value = "relu";
    // Child controls retain their values when their fieldset is hidden/disabled.
    const hiddenUnits = { value: "64, 32" };
    const normalization = { checked: true };
    elements["fully-connected-fields"].children = [hiddenUnits, normalization];
    if (includeResult) {
        elements["generated-config"] = { hidden: false };
        elements["server-error"] = { hidden: false };
    }
    let change;
    elements.network_type.addEventListener = (event, callback) => {
        assert.equal(event, "change");
        change = callback;
    };
    vm.runInNewContext(script, { document: { getElementById: id => elements[id] ?? null } });
    return { elements, hiddenUnits, normalization, select(type) {
        elements.network_type.value = type;
        change();
    } };
}

function assertMode(elements, isFullyConnected) {
    assert.equal(elements["fully-connected-fields"].hidden, !isFullyConnected);
    assert.equal(elements["fully-connected-fields"].disabled, !isFullyConnected);
    assert.equal(elements["cnn-fields"].hidden, isFullyConnected);
    assert.equal(elements["cnn-status"].hidden, isFullyConnected);
    // Hidden CNN draft values are submitted with FC forms, so an error does not erase them.
    assert.equal(elements["cnn-fields"].disabled, false);
    assert.equal(elements["generate-button"].disabled, !isFullyConnected);
    assert.equal(elements.input_shape.required, isFullyConnected);
    assert.equal(elements.output_shape.required, isFullyConnected);
    assert.equal(elements.input_shape.hidden, false);
    assert.equal(elements.output_shape.hidden, false);
    assert.equal(elements["fc-input-help"].hidden, !isFullyConnected);
    assert.equal(elements["cnn-input-help"].hidden, isFullyConnected);
    assert.equal(elements.input_shape.placeholder, isFullyConnected ? "e.g. 34" : "e.g. 128, 128, 3");
}

test("switch FC to CNN and back, preserving values and hiding stale output", () => {
    const view = page("fully_connected");
    assertMode(view.elements, true);
    assert.equal(view.elements["generated-config"].hidden, false);
    view.select("cnn");
    assertMode(view.elements, false);
    assert.equal(view.elements["generated-config"].hidden, true);
    assert.equal(view.elements["server-error"].hidden, true);
    view.select("fully_connected");
    assertMode(view.elements, true);
    assert.equal(view.elements.input_shape.value, "34");
    assert.equal(view.elements.output_shape.value, "10, 20");
    assert.equal(view.hiddenUnits.value, "64, 32");
    assert.equal(view.normalization.checked, true);
    assert.equal(view.elements["generated-config"].hidden, true);
});

test("initialize a server-rendered CNN selection and switch back without results", () => {
    const view = page("cnn", false);
    assertMode(view.elements, false);
    view.select("fully_connected");
    assertMode(view.elements, true);
});

test("crafted unknown client selection cannot enable generation", () => {
    const view = page("fully_connected", false);
    view.select("unknown");
    assert.equal(view.elements["generate-button"].disabled, true);
    assert.equal(view.elements["fully-connected-fields"].disabled, true);
    assert.equal(view.elements["cnn-fields"].hidden, true);
    assert.equal(view.elements["cnn-status"].hidden, true);
});

test("CNN draft and multidimensional input survive switching to FC and back", () => {
    const view = page("cnn", false);
    view.elements.input_shape.value = "128, 128, 3";
    view.select("fully_connected");
    assertMode(view.elements, true);
    view.select("cnn");
    assertMode(view.elements, false);
    assert.equal(view.elements.input_shape.value, "128, 128, 3");
    assert.equal(view.elements.conv_number_filters.value, "8, 16");
    assert.equal(view.elements.conv_kernel_size.value, "3, 3");
    assert.equal(view.elements.conv_pool_size.value, "0, 2");
    assert.equal(view.elements.conv_padding.value, "same");
    assert.equal(view.elements.conv_activation.value, "relu");
});
