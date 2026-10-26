package vn.tvd.plate;

import java.util.Map;
import java.util.Set;

/**
 * Fit OCR text to the Vietnamese plate layout: 2 digits (province) + series (1 letter + optional letter/digit)
 * + 4 or 5 digits, fixing look-alike characters by position (a letter where a digit belongs, and the other way
 * round). The same characters can fit both layouts (30L90860 is car "30L-908.60" or motorbike "30-L9 0860");
 * {@code motorbike} - from the detected vehicle class, since some cars also carry square plates - decides which
 * one is tried first.
 */
public final class PlateNormalizer {
    static final Map<Character, Character> TO_DIGIT = Map.ofEntries(
            Map.entry('O', '0'), Map.entry('Q', '0'), Map.entry('D', '0'), Map.entry('U', '0'),
            Map.entry('I', '1'), Map.entry('L', '1'), Map.entry('T', '1'), Map.entry('Z', '2'),
            Map.entry('J', '3'), Map.entry('A', '4'), Map.entry('S', '5'), Map.entry('G', '6'), Map.entry('B', '8'));
    static final Map<Character, Character> TO_LETTER = Map.of(
            '0', 'D', '1', 'T', '2', 'Z', '4', 'A', '5', 'S', '6', 'G', '8', 'B');
    /** Letters used in Vietnamese plate series (no I, J, O, Q, R, W). */
    static final Set<Character> SERIES_LETTERS = Set.of(
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z');

    private PlateNormalizer() {
    }

    public static String normalize(String raw, boolean motorbike) {
        String s = raw.toUpperCase().replaceAll("[^A-Z0-9]", "");
        if (s.length() < 7 || s.length() > 10) {
            return null;
        }
        String best = null;
        int bestChanges = Integer.MAX_VALUE;
        // Try series of length 1 (51A-12345) and 2 (29B1-23456 / 51LD-12345)
        int[] order = motorbike ? new int[] {2, 1} : new int[] {1, 2};
        for (int seriesLen : order) {
            String num = s.substring(2 + seriesLen);
            if (num.length() != 4 && num.length() != 5) {
                continue;
            }
            String prov = toDigits(s.substring(0, 2));
            char first = TO_LETTER.getOrDefault(s.charAt(2), s.charAt(2));
            String series = first + (seriesLen == 2 ? String.valueOf(s.charAt(3)) : "");
            String digits = toDigits(num);
            if (!isDigits(prov) || !SERIES_LETTERS.contains(first) || !isDigits(digits)) {
                continue;
            }
            if (seriesLen == 2 && !(Character.isDigit(s.charAt(3)) || SERIES_LETTERS.contains(s.charAt(3)))) {
                continue;
            }
            String tail = digits.length() == 5 ? digits.substring(0, 3) + "." + digits.substring(3) : digits;
            // Motorbike series end in a digit and are written "29-B1 256.62"; cars "51A-123.45" / "50LD-004.54"
            String cand = Character.isDigit(series.charAt(series.length() - 1))
                    ? prov + "-" + series + " " + tail : prov + series + "-" + tail;
            // Prefer the reading that needed fewer substitutions
            String flat = cand.replaceAll("[^A-Z0-9]", "");
            int changes = 0;
            for (int i = 0; i < Math.min(s.length(), flat.length()); i++) {
                if (s.charAt(i) != flat.charAt(i)) {
                    changes++;
                }
            }
            if (changes < bestChanges) {
                bestChanges = changes;
                best = cand;
            }
        }
        return best;
    }

    private static String toDigits(String s) {
        StringBuilder b = new StringBuilder();
        for (char c : s.toCharArray()) {
            b.append(TO_DIGIT.getOrDefault(c, c));
        }
        return b.toString();
    }

    private static boolean isDigits(String s) {
        return !s.isEmpty() && s.chars().allMatch(Character::isDigit);
    }
}
