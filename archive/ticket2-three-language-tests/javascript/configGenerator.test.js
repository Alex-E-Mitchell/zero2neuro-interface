const test = require("node:test");
const assert = require("node:assert/strict");

const {
  validateHiddenUnits,
  generateNetworkConfig,
} = require("./configGenerator");

test("validates hidden layer sizes", () => {
  assert.equal(validateHiddenUnits([64, 32]), true);
  assert.equal(validateHiddenUnits([128]), true);

  assert.equal(validateHiddenUnits([0, 32]), false);
  assert.equal(validateHiddenUnits([-4, 32]), false);
  assert.equal(validateHiddenUnits([]), false);
  assert.equal(validateHiddenUnits([64, 3.5]), false);
});

test("generates network configuration text", () => {
  const config = generateNetworkConfig({
    networkType: "fully_connected",
    hiddenUnits: [64, 32],
    hiddenActivation: "elu",
    outputShape: 1,
    outputActivation: "elup1",
  });

  assert.match(config, /--network_type=fully_connected/);
  assert.match(config, /--number_hidden_units\n64\n32/);
  assert.match(config, /--hidden_activation=elu/);
  assert.match(config, /--output_shape\n1/);
  assert.match(config, /--output_activation=elup1/);
});
