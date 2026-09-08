public class ConfigGeneratorTest {

        public static void main(String[] args) {
                runTest(
                                ConfigGenerator.validateHiddenUnits(new int[] { 64, 32 }),
                                true,
                                "Valid hidden units [64, 32]");

                runTest(
                                ConfigGenerator.validateHiddenUnits(new int[] { 128 }),
                                true,
                                "Valid hidden units [128]");

                runTest(
                                ConfigGenerator.validateHiddenUnits(new int[] { 0, 32 }),
                                false,
                                "Zero should be invalid");

                runTest(
                                ConfigGenerator.validateHiddenUnits(new int[] { -4, 32 }),
                                false,
                                "Negative values should be invalid");

                runTest(
                                ConfigGenerator.validateHiddenUnits(new int[] {}),
                                false,
                                "Empty array should be invalid");
        }

        private static void runTest(
                        boolean actual,
                        boolean expected,
                        String description) {
                if (actual == expected) {
                        System.out.println("PASS: " + description);
                } else {
                        System.out.println("FAIL: " + description);
                }
        }
}