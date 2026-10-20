const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const script = fs.readFileSync(path.join(__dirname, "../../static/dropout_validation.js"), "utf8");

function page(initialValue = "") {
    const inputs = {};
    for (const id of ["dropout", "dropout_input"]) {
        const events = {};
        inputs[id] = {
            value: initialValue,
            get valueAsNumber() { return this.value === "" ? NaN : Number(this.value); },
            validity: { badInput: false },
            validationMessage: "",
            setCustomValidity(message) { this.validationMessage = message; },
            addEventListener(name, callback) { events[name] = callback; },
            dispatch(name) { events[name](); },
        };
    }
    vm.runInNewContext(script, { document: { getElementById: id => inputs[id] } });
    return inputs;
}

for (const [id, label] of [["dropout", "Dropout rate"], ["dropout_input", "Input dropout rate"]]) {
    test(`${id}: exclusive range message on input and native invalid events`, () => {
        const input = page()[id];
        for (const value of ["1", "1.0", "1.1", "-0.1"]) {
            for (const event of ["input", "invalid"]) {
                input.setCustomValidity("");
                input.value = value;
                input.dispatch(event);
                assert.equal(input.validationMessage, `${label} must be at least 0 and less than 1.`);
            }
        }
    });

    test(`${id}: correcting or clearing a value clears the error`, () => {
        const input = page("1")[id];
        for (const value of ["0", "0.2", "0.9999999999999999", ""]) {
            assert.notEqual(input.validationMessage, "");
            input.value = value;
            input.dispatch("input");
            assert.equal(input.validationMessage, "");
            input.value = "1";
            input.dispatch("input");
        }
    });

    test(`${id}: incomplete numeric input does not clear the error as if blank`, () => {
        const input = page()[id];
        input.validity.badInput = true;
        input.dispatch("input");
        assert.equal(input.validationMessage, `${label} must be at least 0 and less than 1.`);
        input.validity.badInput = false;
        input.dispatch("input");
        assert.equal(input.validationMessage, "");
    });
}
