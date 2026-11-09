"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const source = fs.readFileSync(path.join(__dirname, "../../static/form_persistence.js"), "utf8");

function field(name, value = "", type = "text", more = {}) {
    return { name, value, type, tagName: "INPUT", checked: false, disabled: false, ...more };
}
function makePage(page, elements, storage, hooks = {}) {
    const listeners = {};
    const form = {
        dataset: { draftKey: page }, elements,
        addEventListener(event, callback) { listeners[event] = callback; },
        querySelector() { return null; },
    };
    const localStorage = {
        getItem(key) { return storage[key] ?? null; },
        setItem(key, value) { storage[key] = value; },
    };
    const context = { document: { querySelector: () => form }, window: { localStorage }, ...hooks };
    vm.runInNewContext(source, context);
    return { listeners, elements };
}
const store = {};
let page = makePage("network", [
    field("input_shape", "34"),
    field("network_type", "fully_connected", "select-one", { tagName: "SELECT" }),
    field("batch_normalization", "on", "checkbox", { checked: true }),
    field("conv_kernel_size", "3, 3", "text", { disabled: true }),
], store);
page.listeners.input();
assert.equal(JSON.parse(store["zero2neuro:draft:v1:network"]).input_shape, "34");
assert.equal(JSON.parse(store["zero2neuro:draft:v1:network"]).conv_kernel_size, "3, 3");
let updates = 0;
page = makePage("network", [
    field("input_shape"), field("network_type", "cnn", "select-one", { tagName: "SELECT" }),
    field("batch_normalization", "on", "checkbox"), field("conv_kernel_size"),
], store, { updateNetworkType() { updates++; } });
assert.equal(page.elements[0].value, "34");
assert.equal(page.elements[1].value, "fully_connected");
assert.equal(page.elements[2].checked, true);
assert.equal(page.elements[3].value, "3, 3");
assert.equal(updates, 1);

const data = makePage("data", [field("data_files", "a.csv"), field("data_n_folds", "5", "text", { disabled: true })], store);
data.listeners.change();
assert.equal(JSON.parse(store["zero2neuro:draft:v1:data"]).data_n_folds, "5");
assert.equal(JSON.parse(store["zero2neuro:draft:v1:network"]).input_shape, "34");
let dataSync = 0;
const restored = makePage("data", [field("data_files"), field("data_n_folds")], store, { syncDataForm() { dataSync++; } });
assert.equal(restored.elements[0].value, "a.csv");
assert.equal(restored.elements[1].value, "5");
assert.equal(dataSync, 1);

const exp = makePage("experiment", [
    field("metrics", "mse", "checkbox", { checked: true }),
    field("metrics", "mae", "checkbox"),
    field("early_stopping", "on", "checkbox", { checked: true }),
], store);
exp.listeners.submit();
const exp2 = makePage("experiment", [
    field("metrics", "mse", "checkbox"),
    field("metrics", "mae", "checkbox", { checked: true }),
    field("early_stopping", "on", "checkbox"),
], store);
assert.equal(exp2.elements[0].checked, true);
assert.equal(exp2.elements[1].checked, false);
assert.equal(exp2.elements[2].checked, true);

const form = { dataset: { draftKey: "network" }, elements: [field("input_shape", "42")], listeners: {}, addEventListener(event, cb) { this.listeners[event] = cb; }, querySelector() { return null; } };
const unavailableContext = { document: { querySelector: () => form }, window: { get localStorage() { throw new Error("storage blocked"); } } };
vm.runInNewContext(source, unavailableContext);
form.listeners.input(); // Storage failures should not throw or stop the form.
console.log("Form persistence tests passed (navigation, drafts, disabled fields, checkboxes, submit).");

// Clearing is local to the selected tab and returns the page to its GET defaults.
for (const currentTab of ["network", "data", "experiment"]) {
    const events = {};
    const form = {
        dataset: { draftKey: currentTab },
        elements: [field("a", "edited")],
        addEventListener(event, callback) { events[event] = callback; },
        querySelector(selector) {
            assert.equal(selector, "[data-clear-inputs]");
            return { addEventListener(event, callback) { events[`clear-${event}`] = callback; } };
        },
    };
    const key = `zero2neuro:draft:v1:${currentTab}`;
    const otherKey = `zero2neuro:draft:v1:other`;
    const saved = { [key]: JSON.stringify({ a: "edited" }), [otherKey]: "preserve" };
    let navigated = null;
    let approve = false;
    vm.runInNewContext(source, {
        document: { querySelector() { return form; } },
        window: {
            localStorage: {
                getItem(k) { return saved[k] ?? null; },
                setItem(k, v) { saved[k] = v; },
                removeItem(k) { delete saved[k]; },
            },
            confirm() { return approve; },
            location: { pathname: `/${currentTab}`, assign(target) { navigated = target; } },
        },
    });
    events["clear-click"]();
    assert.ok(saved[key]);
    assert.equal(navigated, null);
    approve = true;
    events["clear-click"]();
    assert.equal(saved[key], undefined);
    assert.equal(saved[otherKey], "preserve");
    assert.equal(navigated, `/${currentTab}`);
}
console.log("Clear Inputs tests passed (scoping, cancellation, fresh GET).");
