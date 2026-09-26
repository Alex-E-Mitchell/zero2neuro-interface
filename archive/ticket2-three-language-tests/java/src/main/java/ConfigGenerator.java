public class ConfigGenerator {

    public static boolean validateHiddenUnits(int[] units) {
        if (units == null || units.length == 0) {
            return false;
        }

        for (int unit : units) {
            if (unit <= 0) {
                return false;
            }
        }

        return true;
    }

    public static String generateNetworkConfig(
            String networkType,
            int[] hiddenUnits,
            String hiddenActivation,
            int outputShape,
            String outputActivation) {

        if (!validateHiddenUnits(hiddenUnits)) {
            throw new IllegalArgumentException(
                    "Hidden layer sizes must be positive integers."
            );
        }

        if (outputShape <= 0) {
            throw new IllegalArgumentException(
                    "Output shape must be a positive integer."
            );
        }

        StringBuilder config = new StringBuilder();

        config.append("--network_type=").append(networkType).append("\n");
        config.append("--number_hidden_units\n");

        for (int unit : hiddenUnits) {
            config.append(unit).append("\n");
        }

        config.append("--hidden_activation=")
              .append(hiddenActivation)
              .append("\n");

        config.append("--output_shape\n")
              .append(outputShape)
              .append("\n");

        config.append("--output_activation=")
              .append(outputActivation)
              .append("\n");

        return config.toString();
    }
}
