package vn.tvd.rules;

public record Violation(String type, Track track, String detail, double t, Double speedKmh) {
    public Violation(String type, Track track, String detail, double t) {
        this(type, track, detail, t, null);
    }
}
