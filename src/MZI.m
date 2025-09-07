function MZI = MZI(theta, phi_u, phi_d)


pref = -1j * exp(-1j * theta/2);
s = sin(theta/2);
c = cos(theta/2);

MZI = pref * [ s*exp(-1j*phi_u),    c*exp(-1j*phi_d);
              c*exp(-1j*phi_u),   -s*exp(-1j*phi_d) ];

end
