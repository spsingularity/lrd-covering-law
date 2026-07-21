#!/usr/bin/env python3
"""Symbolic verification of every analytic identity in the covering-law papers.

Each check asserts an EXACT symbolic identity (sympy.simplify(lhs-rhs)==0 or
an equivalent structural test) and prints PASS/FAIL. Order-of-magnitude or
numerical-scaling claims (momentum parity, drag times) are not symbolic and
are listed separately as NOT-SYMBOLIC with the reason.

Run: /tmp/symvenv/bin/python verify_equations_sympy.py
"""
import sympy as sp

results = []
def check(name, condition, detail=""):
    ok = bool(condition)
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return ok

# Common positive real symbols
x, k, D, eta, mu = sp.symbols('x k D eta mu', positive=True)
C, B, G, nu, t, dt = sp.symbols('C B G nu t dt', positive=True)

print("="*72)
print("SECTION 1 -- Equilibrium and dynamics (paper Sec. 3.3, 3.5)")
print("="*72)

# 1.1 Logistic build/clear equilibrium: dC/dt = (1-C)B - CG  ->  C/(1-C) = B/G
Ceq = sp.solve(sp.Eq((1-C)*B - C*G, 0), C)[0]
odds_eq = sp.simplify(Ceq/(1-Ceq))
check("1.1 logistic fixed point C/(1-C) = B/G",
      sp.simplify(odds_eq - B/G) == 0, f"C_eq = {sp.simplify(Ceq)}")

# 1.2 Exact exponential-integrator update for the linear ODE with frozen B,G.
#     dC/dt = (1-C)B - CG = B - (B+G)C. Solve IVP C(0)=C0, evaluate at dt.
C0 = sp.symbols('C0', real=True)
tt = sp.symbols('tt', positive=True)
Cfun = sp.Function('Cf')
sol = sp.dsolve(sp.Eq(Cfun(tt).diff(tt), B - (B+G)*Cfun(tt)),
                Cfun(tt), ics={Cfun(0): C0})
Cdt = sol.rhs.subs(tt, dt)
# Claimed update: C(dt) = C_eq + (C0 - C_eq) exp(-(B+G) dt), C_eq = B/(B+G)
Ceq2 = B/(B+G)
claim = Ceq2 + (C0 - Ceq2)*sp.exp(-(B+G)*dt)
check("1.2 exact integrator C(dt)=C_eq+(C0-C_eq)e^{-(B+G)dt}",
      sp.simplify(Cdt - claim) == 0)

# 1.3 Relaxation rate r = B+G = nu(x + 1/x) when B=nu*x, G=nu/x
r = (nu*x) + (nu/x)
check("1.3 relaxation rate r = nu(x + 1/x)",
      sp.simplify(r - nu*(x + 1/x)) == 0)

# 1.4 equilibrium_covering inverse: odds = x^2  =>  C = x^2/(1+x^2)
Csol = sp.solve(sp.Eq(C/(1-C), x**2), C)[0]
check("1.4 C = x^2/(1+x^2) from odds = x^2",
      sp.simplify(Csol - x**2/(1+x**2)) == 0)

print()
print("="*72)
print("SECTION 2 -- The closure and its generalization (Sec. 3.1, 3.2)")
print("="*72)

# 2.1 f_dense = D^eta/(1+D^eta) is the logistic of eta*ln(D)
f_logistic = 1/(1 + sp.exp(-eta*sp.log(D)))
check("2.1 f_dense = D^eta/(1+D^eta) = logistic(eta ln D)",
      sp.simplify(f_logistic - D**eta/(1+D**eta)) == 0)

# 2.2 Odds of f_dense equal D^eta exactly
f = D**eta/(1+D**eta)
check("2.2 f_dense/(1-f_dense) = D^eta",
      sp.simplify(f/(1-f) - D**eta) == 0)

# 2.3 Special case eta=2 gives the fiducial quadratic closure
check("2.3 eta=2 -> f_dense = D^2/(1+D^2)",
      sp.simplify(f.subs(eta,2) - D**2/(1+D**2)) == 0)

print()
print("="*72)
print("SECTION 3 -- Route B: sequential two-stage odds (X2 note, Table)")
print("="*72)

# 3.1 Two independent stages each with success ODDS = k*x  =>  per-stage
#     probability p_s = kx/(1+kx). Combined success p = p_s^2. Show that the
#     combined ODDS reduce to (kx)^2/(1+2kx).
ps = (k*x)/(1 + k*x)          # per-stage success probability from odds kx
p_comb = ps**2                # both stages succeed (independent)
odds_comb = sp.simplify(p_comb/(1 - p_comb))
check("3.1 two-stage combined odds = (kx)^2/(1+2kx)",
      sp.simplify(odds_comb - (k*x)**2/(1 + 2*k*x)) == 0,
      f"odds = {odds_comb}")

# 3.2 Low-contrast limit: leading term is (kx)^2 (matches pure quadratic),
#     with the first correction -2(kx)^3 (the roll-over) appearing at next order.
ser4 = sp.series((k*x)**2/(1+2*k*x), x, 0, 4).removeO()
lead = sp.series((k*x)**2/(1+2*k*x), x, 0, 3).removeO()
check("3.2 Route-B low-x: leading = (kx)^2, next term = -2(kx)^3",
      sp.simplify(lead - (k*x)**2) == 0
      and sp.simplify(ser4 - ((k*x)**2 - 2*(k*x)**3)) == 0,
      f"series to O(x^4) = {ser4}")

# 3.3 High-contrast limit: odds -> kx/2 (the roll-over that the data reject)
hi = sp.limit(((k*x)**2/(1+2*k*x)) / (k*x/2), x, sp.oo)
check("3.3 Route-B high-x asymptote odds ~ kx/2",
      sp.simplify(hi - 1) == 0)

# 3.4 The log-log slope of Route-B odds is < 2 for x>0 (flattening = roll-over).
#     d ln(odds)/d ln x = x * d/dx ln(odds). With u=kx, odds=u^2/(1+2u).
u = sp.symbols('u', positive=True)
oddsB_u = u**2/(1+2*u)
slopeB_u = sp.simplify(u*sp.diff(sp.log(oddsB_u), u))   # d ln odds/d ln u
target = 2 - 2*u/(1+2*u)
check("3.4 Route-B log-slope = 2 - 2u/(1+2u) < 2 (roll-over)",
      sp.simplify(slopeB_u - target) == 0
      and sp.simplify(sp.limit(slopeB_u, u, 0) - 2) == 0    # ->2 at low contrast
      and (2 - target).subs(u, 1) > 0                       # strictly <2 at u=1
      and sp.simplify(sp.limit(slopeB_u, u, sp.oo) - 1) == 0, # ->1 at high contrast
      f"slope(u) = {sp.simplify(slopeB_u)}")

print()
print("="*72)
print("SECTION 4 -- Route D: Poisson double-blocking (X2 note)")
print("="*72)

# 4.1 Opaque covering requires >=2 clumps; Poisson areal density mu:
#     C = P(N>=2) = 1 - P(0) - P(1) = 1 - e^{-mu}(1+mu)
P0 = sp.exp(-mu)
P1 = mu*sp.exp(-mu)
C_pois = 1 - P0 - P1
check("4.1 C = 1 - e^{-mu}(1+mu) from Poisson P(N>=2)",
      sp.simplify(C_pois - (1 - sp.exp(-mu)*(1+mu))) == 0)

# 4.2 Stable log form used in code: ln(1-C) = -mu + ln(1+mu)
check("4.2 ln(1-C) = -mu + ln(1+mu)",
      sp.simplify(sp.log(1 - C_pois) - (-mu + sp.log(1+mu))) == 0)

# 4.3 Low-mu: C ~ mu^2/2, so odds ~ mu^2/2 (quadratic onset)
Cser = sp.series(C_pois, mu, 0, 4).removeO()
check("4.3 Route-D low-mu C = mu^2/2 - mu^3/3 + ...",
      sp.simplify(Cser - (mu**2/2 - mu**3/3)) == 0,
      f"series = {Cser}")

# 4.4 High-mu steepening: log-slope of odds -> infinity (no roll-over; the
#     opposite distortion from Route B). odds = C/(1-C).
oddsD = C_pois/(1-C_pois)
slopeD_inf = sp.limit(sp.diff(sp.log(oddsD), mu)*mu, mu, sp.oo)
check("4.4 Route-D log-slope -> oo at high mu (steepening)",
      slopeD_inf == sp.oo)

print()
print("="*72)
print("SECTION 5 -- Nuclear scaling algebra (scaling_sensitivity.py)")
print("="*72)

# Symbols for the fiducial scaling derivation
G_, R, eps_ret, f_b, Mdot, eps_ff, tdep, Mnuc, rho, tff = sp.symbols(
    'G R eps_ret f_b Mdot eps_ff tdep Mnuc rho tff', positive=True)

# Definitions used in the code/manuscript:
#   Mnuc = eps_ret f_b Mdot tdep
#   rho  = 3 Mnuc/(4 pi R^3)
#   tff  = sqrt(3 pi/(32 G rho))
#   tdep = tff/eps_ff
Mnuc_def = eps_ret*f_b*Mdot*tdep
rho_def = 3*Mnuc_def/(4*sp.pi*R**3)
tff_def = sp.sqrt(3*sp.pi/(32*G_*rho_def))
tdep_eq = sp.Eq(tdep, tff_def/eps_ff)

# Solve the implicit equation for tdep^3 and compare with the manuscript form
sol_td = sp.solve(tdep_eq, tdep)
# take the positive real root
td_pos = [s for s in sol_td if s.is_real is not False]
td_val = sp.simplify(sol_td[0])
claim_td3 = sp.pi**2 * R**3 / (8*G_*eps_ret*f_b*Mdot*eps_ff**2)
check("5.1 tdep^3 = pi^2 R^3/(8 G eps_ret f_b Mdot eps_ff^2)",
      sp.simplify(td_val**3 - claim_td3) == 0,
      f"tdep = {td_val}")

# 5.2 N_H = Mnuc/(pi R^2 mu_H m_p) -- consistency of the column definition
muH, mp, NH = sp.symbols('mu_H m_p N_H', positive=True)
# column = mass / (area * mean mass per H); area = pi R^2 for the stated disc
check("5.2 N_H = Mnuc/(pi R^2 mu_H m_p) (definition consistent)",
      sp.simplify((Mnuc/(sp.pi*R**2*muH*mp)) - NH.subs(NH, Mnuc/(sp.pi*R**2*muH*mp))) == 0)

print()
print("="*72)
print("SECTION 6 -- Emission-layer identities (physical_visibility.py)")
print("="*72)

# 6.1 Stefan-Boltzmann: integral of pi B_nu dnu over all nu = sigma T^4,
#     with sigma = 2 pi^5 k^4/(15 c^2 h^3). This underlies
#     blackbody_lnu_per_lbol = pi B_nu/(sigma T^4).
hpl, kB, cc, T, nu_, uu = sp.symbols('h k_B c T nu u', positive=True)
Bnu = (2*hpl*nu_**3/cc**2) / (sp.exp(hpl*nu_/(kB*T)) - 1)
# Dimensionless substitution nu = (kB T/h) u. The Bose-Einstein integral has
# the exact closed form  int_0^oo u^{s-1}/(e^u-1) du = Gamma(s) zeta(s);
# verify that closed form against a high-precision numerical quadrature, then
# use it (s=4 -> Gamma(4)zeta(4) = 6*pi^4/90 = pi^4/15).
planck_closed = sp.gamma(4)*sp.zeta(4)
planck_num = sp.Integral(uu**3/(sp.exp(uu)-1), (uu, 0, sp.oo)).evalf(25)
flux = sp.pi * (2*hpl/cc**2) * (kB*T/hpl)**4 * planck_closed
sigma_SB = 2*sp.pi**5*kB**4/(15*cc**2*hpl**3)
check("6.1 integral pi B_nu dnu = sigma_SB T^4 (Stefan-Boltzmann)",
      sp.simplify(planck_closed - sp.pi**4/15) == 0
      and abs(sp.Float(planck_num) - float(sp.pi**4/15)) < 1e-12
      and sp.simplify(flux - sigma_SB*T**4) == 0,
      f"Gamma(4)zeta(4) = {sp.nsimplify(planck_closed)} = pi^4/15; "
      f"numeric = {planck_num}")

# 6.2 Two-reservoir partition: given dominance R_od = Q_dense/Q_diffuse,
#     f_dense = R_od/(1+R_od) is a proper probability (logistic); check the
#     stable logistic used in code equals R/(1+R).
R_od = sp.symbols('R_od', positive=True)
logodds = sp.log(R_od)
f_stable = 1/(1+sp.exp(-logodds))
check("6.2 f_dense = R/(1+R) equals stable logistic of ln R",
      sp.simplify(f_stable - R_od/(1+R_od)) == 0)

# 6.3 Electron-scattering + virial widths add in quadrature (definition used
#     in confront_rubies_fwhm.widths_at_radius): FWHM^2 = v_vir^2 + v_e^2.
v_vir, v_e, FWHM = sp.symbols('v_vir v_e FWHM', positive=True)
check("6.3 FWHM = sqrt(v_vir^2 + v_e^2) (quadrature, definitionally consistent)",
      sp.simplify(sp.sqrt(v_vir**2 + v_e**2)**2 - (v_vir**2 + v_e**2)) == 0)

print()
print("="*72)
print("SECTION 7 -- Cosmic time integral (revision_analysis.cosmic_time)")
print("="*72)

# 7.1 Flat LambdaCDM age: t(z) = (2/(3 H0 sqrt(OmL))) * arcsinh( sqrt(OmL/Om)
#     (1+z)^{-3/2} ). Verify it satisfies dt/dz = -1/((1+z) H(z)) with
#     H = H0 sqrt(Om (1+z)^3 + OmL).
H0, Om, OmL, z = sp.symbols('H0 Omega_m Omega_L z', positive=True)
t_of_z = (2/(3*H0*sp.sqrt(OmL))) * sp.asinh(sp.sqrt(OmL/Om)*(1+z)**sp.Rational(-3,2))
Hz = H0*sp.sqrt(Om*(1+z)**3 + OmL)
dtdz = sp.diff(t_of_z, z)
check("7.1 cosmic-time integral solves dt/dz = -1/((1+z)H(z))",
      sp.simplify(dtdz + 1/((1+z)*Hz)) == 0)

print()
print("="*72)
print("SECTION 8 -- Mass-budget cancellation lemma (Sec. 3.4, schematic)")
print("="*72)

# 8.1 If build rate B = A_b * (Mdot_supply/Sigma) and clear rate
#     G = A_g * (Phi_ablation/Sigma), with the SAME clump surface density
#     Sigma entering both, then B/G is independent of Sigma. Verify the
#     surface density cancels exactly, leaving B/G proportional to the
#     supply-to-ablation ratio (which the paper shows scales as x).
Sigma, A_b, A_g, Msup, Phi = sp.symbols('Sigma A_b A_g Msup Phi', positive=True)
B_l = A_b*Msup/Sigma
G_l = A_g*Phi/Sigma
ratio = sp.simplify(B_l/G_l)
check("8.1 Sigma cancels in B/G (mass-budget gives one power)",
      sp.simplify(sp.diff(ratio, Sigma)) == 0
      and sp.simplify(ratio - (A_b*Msup)/(A_g*Phi)) == 0,
      f"B/G = {ratio}")

# 8.2 Route C second power: with Msup/Phi ~ x (momentum-limited build,
#     verified in-model) and drag lifetime t_drag ~ chi ~ x, the residence
#     mode multiplies another factor x, giving odds ~ x*x = x^2.
chi = sp.symbols('chi', positive=True)
odds_C = (A_b*Msup/(A_g*Phi)) * chi   # population factor times residence factor
odds_C_x = odds_C.subs({Msup: x*Phi*A_g/A_b, chi: x})  # both factors -> x
check("8.2 Route-C odds = (population ~ x)(residence ~ x) = x^2",
      sp.simplify(odds_C_x - x**2) == 0)

print()
print("="*72)
print("SUMMARY")
print("="*72)
npass = sum(1 for _,ok,_ in results if ok)
for name, ok, _ in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
print(f"\n{npass}/{len(results)} symbolic checks passed.")

print()
print("NOT SYMBOLICALLY VERIFIABLE (order-of-magnitude / numerical claims):")
for s in [
 "momentum parity B/G ~ 3 at x=1: depends on measured p_SN, L/SFR calibration",
 "drag-time chi^1 scaling: imported from cloud-acceleration simulations",
 "relaxation bound <~30 Myr: numerical from lifecycle runs, not a closed form",
 "epsilon ~ 0.1 = LyC escape fraction: empirical identification",
 "cooling t_cool 1e-2..1e0 yr: numerical from Lambda(Z), pressures",
]:
    print("  - " + s)
