function D = ztoD(z)

c  = 2.9979e8;          %[m/s]
Ho = 67.74;                %Hubble's constant, [km/s/Mpc]
omegaM = 0.3089;         %WMAP 7 maximum likelihood values
omegaL = 0.6911;
omegak = 1-omegaM-omegaL;
invEz = @(x) 1./sqrt(omegaM*(1+x).^3+omegak*(1+x).^2+omegaL);   %x=z

D=(1+z)*(c/1e6)/Ho*integral(invEz,0,z);

end