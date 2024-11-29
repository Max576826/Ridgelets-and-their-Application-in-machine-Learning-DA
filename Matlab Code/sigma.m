function z = sigma(x)
    %% Initialization of the output variable and parameters
    z = 0;
    a = -1;
    b = 1;
    d = b - a;
    lam = 0.4;
    n = 1000;
    C = 10;
    x = single(x);

    %% Helper functions
    
    % Helper function for boundary control
    function y = h(z)
        y = 1 - min(0.5, lam) / (1 + log(z - d + 1));
    end

    % Calkin-Wilf sequence, computes rational number on position k
    function y = CW(k)
        y = 1;
        k = k - 1;
        [N, D] = rat(y);
        for i = 1:k
            frac = 1 / (2 * floor(y) - y + 1);
            [N, D] = rat(frac);
            y = N / D;
        end
    end

    % Function for rational number enumeration
    function y = r(k)
        if k == 0
            y = 0;
        elseif mod(k, 2) == 0
            y = CW(round(k / 2));
        else
            y = -CW(round((k + 1) / 2));
        end
    end

    % Enumeration of all monic polynomials with rational coefficients
    function y = u(k)
        p = 1;
        m = cfrac(CW(k), 100);
        if k == 1
            p = 1;
        elseif m(1) == fix(m(1)) && (m(1) ~= 1)
            p = poly([1, r(CW(k) - 2)]);
        elseif length(m) == 2
            p = poly([1, r(m(2) - 2), r(m(1))]);
        elseif length(m) > 2
            plist = [r(m(1))];
            for i = 2:(length(m) - 1)
                plist = [plist, r(m(i) - 1)];
            end
            plist = [plist, r(m(end) - 2), 1];
            plist = fliplr(plist);
            p = poly(plist);
        end
        y = p;
    end

    % Calculation of B1
    function B1 = B1(lst)
        B1 = lst(1);
        for i = 1:length(lst)
            B1 = B1 + (lst(i) - abs(lst(i))) / 2;
        end
    end

    % Calculation of B2
    function B2 = B2(lst)
        B2 = lst(1) + 1;
        for i = 1:length(lst)
            B2 = B2 + (lst(i) + abs(lst(i))) / 2;
        end
    end

    % Explicit Calculation of the sequence M_k
    function y = M(k)
        y = h((2 * k + 1) * d);
    end

    % Explicit Calculation of the sequence a_k
    function y = af(k)
        if k == 1
            y = 1 / 2;
        else
            y = ((1 + 2 * M(k)) * B2(u(k)) - (2 + M(k)) * B1(u(k))) / (3 * (B2(u(k)) - B1(u(k))));
        end
    end

    % Explicit Calculation of the sequence b_k
    function y = bf(k)
        if k == 1
            y = h(3 * d) / 2;
        else
            y = (1 - M(k)) / (3 * (B2(u(k)) - B1(u(k))));
        end
    end

    % Calculation for x in [(2k-1)d,2kd]
    function y = presigma(k, x)
        p = u(k);
        y = af(k) + bf(k) * polyval(p, x / d - 2 * k + 1);
    end

    % Helper function for the smooth transition function
    function y = betad(x)
        if x > 0
            y = exp(-1 / x);
        else
            y = 0;
        end
    end

    % Smooth transition function
    function y = beta(a, b, x)
        y = betad(b - x) / (betad(b - x) + betad(x - a));
    end

    % Explicit Calculation of the sequence K_k
    function y = K(k)
        y = (presigma(k, 2 * k * d) + presigma(k, (2 * k + 1) * d)) / 2;
    end

    %% Main Calculation

    % Defining the Intervals based on n
    N =3*n;
    A=zeros(N,2);
    for l = 1:n
    
            starti=1+3*(l-1);
            A(starti,1)=2*l-1;
            A(starti+1,1)=2*l;
            A(starti+2,1)=2*l+0.5;
            A(starti,2)=2*l;
            A(starti+1,2)=2*l+0.5;
            A(starti+2,2)=2*l+1;

    end

    % Stretch the Intervals with a given d
    A=d.*A;

    % Find the interval where x lies in
    Akl = find(A(:,1)<=x);
    Akr = find(A(:,2)>x);
    kl = max(Akl);
    kr = min(Akr);

        
        
    % Determine z based on interval
    if isempty(kl) || isempty(kr)
        z = (1 - betad(d - x)) * (1 + M(1)) / 2;
    else
        intervalType = mod(kl, 3);
        k = ceil(kl / 3);
        if intervalType == 2 % presigma interval
            z = presigma(k, x);
        elseif intervalType == 1 % beta interval 1
            eps = (1 - M(k)) / 6;
            delta = min(eps * d / (bf(k) * C), d / 2);
            p = u(k);
            z = K(k) - beta(2 * k * d, 2 * k * d + delta, x) * (K(k) - af(k) - bf(k) * polyval(p, x / d - 2 * k + 1));
        elseif intervalType == 0 % beta interval 2
            p = u(k);
            epsq = (1 - M(k + 1)) / 6;
            deltaq = min(epsq * d / (bf(k + 1) * C), d / 2);
            z = K(k) - (1 - beta((2 * k + 1) * d - deltaq, (2 * k + 1) * d, x)) * (K(k) - af(k) - bf(k) * polyval(p, x / d - 2 * k - 1));
        else
            z = (1 - betad(d - x)) * (1 + M(1)) / 2;
        end
    end
    % Make z single precision for network calculations
    z = single(z);
end

    
