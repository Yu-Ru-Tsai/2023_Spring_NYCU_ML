from utils import gaussian_datagen

if __name__ == '__main__':
    m = 3.0
    s = 5.0
    eps = 1e-1
    square_mean = 0
    m_estimate = 0
    s_estimate = 0
    n = 0
    # Welford's online algorithm
    while (eps < abs(m_estimate - m) or eps < abs(s_estimate - s**2)) and n < 500:

        data = gaussian_datagen(m, s)
        n += 1
        print("Add data point:", data)
        square_mean = (square_mean * (n-1) + data**2) / n
        m_estimate = (m_estimate * (n-1) + data) / n
        s_estimate = square_mean - m_estimate**2  # var(x) = E(x^2) - E(x)^2
        print("Mean=", m_estimate, "Variance=", s_estimate)
    print(f"total sampling {n} point")
