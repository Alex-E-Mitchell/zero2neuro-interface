import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class ConfigGeneratorTest {

    @Test
    void validatesHiddenLayerSizes() {
        assertTrue(
                ConfigGenerator.validateHiddenUnits(
                        new int[]{64, 32}
                )
        );

        assertTrue(
                ConfigGenerator.validateHiddenUnits(
                        new int[]{128}
                )
        );

        assertFalse(
                ConfigGenerator.validateHiddenUnits(
                        new int[]{0, 32}
                )
        );

        assertFalse(
                ConfigGenerator.validateHiddenUnits(
                        new int[]{-4, 32}
                )
        );

        assertFalse(
                ConfigGenerator.validateHiddenUnits(
                        new int[]{}
                )
        );
    }

    @Test
    void generatesNetworkConfigurationText() {
        String config = ConfigGenerator.generateNetworkConfig(
                "fully_connected",
                new int[]{64, 32},
                "elu",
                1,
                "elup1"
        );

        assertTrue(config.contains("--network_type=fully_connected"));
        assertTrue(config.contains("--number_hidden_units\n64\n32"));
        assertTrue(config.contains("--hidden_activation=elu"));
        assertTrue(config.contains("--output_shape\n1"));
        assertTrue(config.contains("--output_activation=elup1"));
    }
}
