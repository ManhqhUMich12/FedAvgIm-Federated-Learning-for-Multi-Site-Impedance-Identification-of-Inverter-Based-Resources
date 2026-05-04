openExample('slcontrol/MeasureInputAdmittanceOutputImpedanceBoostConverterFREExample')
set_param([mdl '/Boost Converter'],...
    'OverrideUsingVariant','block_boost_converter')
in_PRBS = frest.PRBS('Order',14,'NumPeriods',1,'Amplitude',1,'Ts',5e-6);
in_Sinestream = frest.createFixedTsSinestream(5e-06,{200,6e5});
in_Sinestream.Amplitude = 0.5;
in_PRBS.getSimulationTime
in_Sinestream.getSimulationTime
opini = findop(mdl,0.045);
set_param(mdl,'LoadInitialState','on','InitialState','getstatestruct(opini)')
io_Yin(1) = linio([mdl,'/PID Controller'],1,'loopbreak');
io_Yin(2) = linio([mdl,'/Sampling'],1,'loopbreak');
io_Yin(3) = linio([mdl,'/Vin Value'],1,'input');
io_Yin(4) = linio([mdl,'/Rate Transition1'],1,'output');
op_Yin = operpoint(mdl);
srcblksYin = frest.findSources(mdl,io_Yin);
optsYin = frestimateOptions;
optsYin.BlocksToHoldConstant = srcblksYin;
sysestYin_prbs = frestimate(mdl,io_Yin,op_Yin,in_PRBS,optsYin);
io_Zout(1) = linio([mdl,'/PID Controller'],1,'loopbreak');
io_Zout(2) = linio([mdl,'/Sampling'],1,'loopbreak');
io_Zout(3) = linio([mdl,'/Constant'],1,'input');
io_Zout(4) = linio([mdl,'/Rate Transition2'],1,'output');
op_Zout = operpoint(mdl);
srcblksZout = frest.findSources(mdl,io_Zout);
optsZout = frestimateOptions;
optsZout.BlocksToHoldConstant = srcblksZout;
sysestZout_prbs = frestimate(mdl,io_Zout,op_Zout,in_PRBS,optsZout);