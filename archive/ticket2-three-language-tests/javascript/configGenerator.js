function validateHiddenUnits(units) {
  if (!Array.isArray(units) || units.length === 0) {
    return false;
  }

  return units.every((unit) => Number.isInteger(unit) && unit > 0);
}

function generateNetworkConfig({
  networkType,
  hiddenUnits,
  hiddenActivation,
  outputShape,
  outputActivation,
}) {
  if (!validateHiddenUnits(hiddenUnits)) {
    throw new Error("Hidden layer sizes must be positive integers.");
  }

  if (!Number.isInteger(outputShape) || outputShape <= 0) {
    throw new Error("Output shape must be a positive integer.");
  }

  return [
    `--network_type=${networkType}`,
    "--number_hidden_units",
    ...hiddenUnits.map(String),
    `--hidden_activation=${hiddenActivation}`,
    "--output_shape",
    String(outputShape),
    `--output_activation=${outputActivation}`,
    "",
  ].join("\n");
}

module.exports = {
  validateHiddenUnits,
  generateNetworkConfig,
};
