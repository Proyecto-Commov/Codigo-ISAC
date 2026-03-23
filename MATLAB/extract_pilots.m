function [Yp, Xp] = extract_pilots(B_UE, B_tx, P)

    Yp = B_UE(P, :);
    Xp = B_tx(P, :);

end
