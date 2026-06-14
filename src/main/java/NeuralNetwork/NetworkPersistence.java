package NeuralNetwork;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class NetworkPersistence {

    public static void save(NeuralNetwork net, String filePath) {
        try {
            File file = new File(filePath);
            File parent = file.getParentFile();
            if (parent != null && !parent.exists()) {
                parent.mkdirs();
            }

            StringBuilder sb = new StringBuilder();
            sb.append("{\n");

            sb.append("  \"layerSizes\": [");
            for (int i = 0; i < net.layers.length; i++) {
                sb.append(net.layers[i].size);
                if (i < net.layers.length - 1) sb.append(", ");
            }
            sb.append("],\n");

            sb.append("  \"layers\": [\n");
            for (int i = 0; i < net.layers.length; i++) {
                sb.append("    {\n");

                sb.append("      \"biases\": [");
                if (net.layers[i].biases != null) {
                    for (int j = 0; j < net.layers[i].biases.length; j++) {
                        sb.append(net.layers[i].biases[j]);
                        if (j < net.layers[i].biases.length - 1) sb.append(", ");
                    }
                }
                sb.append("],\n");

                sb.append("      \"weights\": [");
                if (net.layers[i].weights != null) {
                    for (int j = 0; j < net.layers[i].weights.length; j++) {
                        sb.append("[");
                        for (int k = 0; k < net.layers[i].weights[j].length; k++) {
                            sb.append(net.layers[i].weights[j][k]);
                            if (k < net.layers[i].weights[j].length - 1) sb.append(", ");
                        }
                        sb.append("]");
                        if (j < net.layers[i].weights.length - 1) sb.append(", ");
                    }
                }
                sb.append("]\n");

                sb.append("    }");
                if (i < net.layers.length - 1) sb.append(",\n");
                else sb.append("\n");
            }
            sb.append("  ]\n");
            sb.append("}");

            Files.write(Paths.get(filePath), sb.toString().getBytes());
        } catch (IOException e) {
            System.err.println("[JSON] Fehler beim Speichern des Netzwerks!");
            e.printStackTrace();
        }
    }

    /**
     * Lädt das Netzwerk aus der JSON.
     * Wenn 'desiredLayers' null oder leer ist, wird die Struktur ungesehen übernommen.
     */
    public static NeuralNetwork load(String filePath, int[] desiredLayers, double eta) {
        File file = new File(filePath);
        if (!file.exists()) {
            return null;
        }

        try {
            String jsonText = new String(Files.readAllBytes(Paths.get(filePath)));
            String compact = jsonText.replaceAll("\\s+", "");

            // 1. Layer-Größen aus der JSON auslesen
            int lsStart = compact.indexOf("\"layerSizes\":[");
            if (lsStart == -1) {
                invalidateFile(file);
                return null;
            }
            lsStart += "\"layerSizes\":".length();
            String lsContent = extractBracketContent(compact, lsStart, '[', ']');
            String[] lsTokens = lsContent.split(",");

            int[] loadedLayers = new int[lsTokens.length];
            for (int i = 0; i < lsTokens.length; i++) {
                loadedLayers[i] = Integer.parseInt(lsTokens[i]);
            }

            boolean acceptAnyModel = (desiredLayers == null || desiredLayers.length == 0);

            if (!acceptAnyModel) {
                if (loadedLayers.length != desiredLayers.length) {
                    invalidateFile(file);
                    return null;
                }
                for (int i = 0; i < desiredLayers.length; i++) {
                    if (loadedLayers[i] != desiredLayers[i]) {
                        invalidateFile(file);
                        return null;
                    }
                }
            }

            // Netzwerk dynamisch mit den ausgelesenen Schichten instanziieren
            NeuralNetwork net = new NeuralNetwork(loadedLayers, eta);

            // 2. Laden der eigentlichen Array-Daten
            int layersArrStart = compact.indexOf("\"layers\":[");
            if (layersArrStart == -1) {
                invalidateFile(file);
                return null;
            }
            layersArrStart += "\"layers\":".length();
            String layersContent = extractBracketContent(compact, layersArrStart, '[', ']');

            List<String> layerObjects = new ArrayList<>();
            int idx = 0;
            while (idx < layersContent.length()) {
                int openBrace = layersContent.indexOf('{', idx);
                if (openBrace == -1) break;
                String objContent = extractBracketContent(layersContent, openBrace, '{', '}');
                layerObjects.add(objContent);
                idx = openBrace + objContent.length() + 2;
            }

            for (int i = 0; i < net.layers.length; i++) {
                String layerObj = layerObjects.get(i);

                int biasStart = layerObj.indexOf("\"biases\":[");
                if (biasStart != -1) {
                    biasStart += "\"biases\":".length();
                    String biasContent = extractBracketContent(layerObj, biasStart, '[', ']');
                    if (!biasContent.trim().isEmpty()) {
                        String[] biasTokens = biasContent.split(",");
                        for (int j = 0; j < biasTokens.length; j++) {
                            net.layers[i].biases[j] = Double.parseDouble(biasTokens[j]);
                        }
                    }
                }

                int weightStart = layerObj.indexOf("\"weights\":[");
                if (weightStart != -1) {
                    weightStart += "\"weights\":".length();
                    String weightContent = extractBracketContent(layerObj, weightStart, '[', ']');
                    if (!weightContent.trim().isEmpty()) {
                        int wIdx = 0;
                        int neuronIdx = 0;
                        while (wIdx < weightContent.length()) {
                            int subOpen = weightContent.indexOf('[', wIdx);
                            if (subOpen == -1) break;
                            String subContent = extractBracketContent(weightContent, subOpen, '[', ']');
                            if (!subContent.trim().isEmpty()) {
                                String[] weightTokens = subContent.split(",");
                                for (int k = 0; k < weightTokens.length; k++) {
                                    net.layers[i].weights[neuronIdx][k] = Double.parseDouble(weightTokens[k]);
                                }
                            }
                            neuronIdx++;
                            wIdx = subOpen + subContent.length() + 2;
                        }
                    }
                }
            }

            return net;

        } catch (Exception e) {
            System.err.println("[JSON] Fehler beim Parsen der JSON-Datei.");
            invalidateFile(file);
            return null;
        }
    }

    private static void invalidateFile(File file) {
        if (file.exists()) {
            file.delete();
        }
    }

    private static String extractBracketContent(String str, int startIdx, char open, char close) {
        int count = 0;
        for (int i = startIdx; i < str.length(); i++) {
            if (str.charAt(i) == open) count++;
            else if (str.charAt(i) == close) {
                count--;
                if (count == 0) return str.substring(startIdx + 1, i);
            }
        }
        return "";
    }
}