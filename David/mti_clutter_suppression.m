function F_clean = mti_clutter_suppression(F)

    [b, a] = butter(2, 0.05, 'high');
    F_clean = filter(b, a, F, [], 2);

end