import types
import logging

import numpy as np
from time import sleep
from instrument import Instrument
from visainstrument import SCPI_Instrument


class Keysight_N5242A(SCPI_Instrument):
    '''
    This is the driver for the Keysight_N5442A PNA-X  (Network Analyzer)

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Keysight_N5442A', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, meas_class='Standard', i_chan=1, **kwargs):
        '''
        Initializes the Keysight_N5242A, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
          meas_class       : PNAX measurement class 'Standard', 'SMC', 'SA', 'SweptIMD', 'IMSpec', 'NFCS'
          i_chan           : channel number on PNAX for this instance of measurement class
        '''
        logging.info(__name__ + ' : Initializing instrument Keysight_N5442A')
        super(Keysight_N5242A, self).__init__(name, address)

        self.add_parameter('chan', type=types.IntType, flags=Instrument.FLAG_GET)
        self._chan = i_chan
        ch = str(i_chan)
        assert meas_class in ['Standard', 'SMC', 'SA', 'SweptIMD', 'IMSpec', 'NFCS'], 'Keysight_N5242A does not support meas_class = ' + meas_class
        self.add_parameter('meas_class', type=types.StringType, flags=Instrument.FLAG_GET)
        self._meas_class = meas_class
        
        self.add_scpi_parameter("start_freq", "SENS"+ch+":FREQ:STAR", "%d", units="Hz", type=types.FloatType, gui_group='sweep') #good
        self.add_scpi_parameter('stop_freq', "SENS"+ch+":FREQ:STOP", "%d", units="Hz", type=types.FloatType, gui_group='sweep') #good
        self.add_scpi_parameter("start_pow", "SOUR"+ch+":POW:STAR", "%d", units="dBm", type=types.FloatType, gui_group='sweep') #good
        self.add_scpi_parameter('stop_pow', "SOUR"+ch+":POW:STOP", "%d", units="dBm", type=types.FloatType, gui_group='sweep')  #good
        self.add_scpi_parameter('center_pow', "SOUR"+ch+":POW:CENT", "%d", units="dBm", type=types.FloatType, gui_group='sweep') #good
        self.add_scpi_parameter('cw_freq',"SENS"+ch+":FREQ:CW", "%d", units='Hz', type=types.FloatType) # now good
        self.add_scpi_parameter('center_freq', "SENS"+ch+":FREQ:CENT", "%d", units="Hz", type=types.FloatType, gui_group='sweep') #good
        self.add_scpi_parameter('span', "SENS"+ch+":FREQ:SPAN", "%d", units="Hz", type=types.FloatType, gui_group='sweep') #good 
        self.add_scpi_parameter('if_bandwidth', "SENS"+ch+":BAND", "%d", units="Hz", type=types.FloatType, gui_group='averaging') #good
        self.add_scpi_parameter('power', "SOUR"+ch+":POW", "%.2f", units="dBm", type=types.FloatType) #good
        self.add_scpi_parameter('power_on', 'OUTP', '%i', type=bool, flags=Instrument.FLAG_GETSET)
        self.add_scpi_parameter('points', "SENS"+ch+":SWE:POIN", "%d", type=types.IntType, gui_group='sweep') #good
        self.add_scpi_parameter('average_factor', "SENS"+ch+":AVER:COUN", "%d", type=types.IntType, gui_group='averaging') #good
        self.add_scpi_parameter('averaging_state', 'SENS'+ch+':AVER', '%i', type=bool, flags=Instrument.FLAG_GETSET, gui_group='averaging')
        self.add_scpi_parameter('averaging_mode', 'SENS'+ch+':AVER:MODE', '%s', type=types.StringType, gui_group='averaging',
                                flags=Instrument.FLAG_GETSET, format_map={'POIN': 'point', 'SWE': 'sweep'})
        # selecting measurements
        self.add_scpi_parameter('meas_select', "CALC"+ch+":PAR:SEL", '%s', type=types.StringType, flags=Instrument.FLAG_GETSET) #new
        self.add_scpi_parameter('meas_select_trace', 'CALC'+ch+':PAR:MNUM', '%d', type=types.IntType, flags=Instrument.FLAG_GETSET)
        self.add_scpi_parameter('meas_class_curr', "SENS"+ch+":CLAS:NAME", '%s', type=types.StringType, flags=Instrument.FLAG_GET) #new
        
        #system
        self.add_scpi_parameter('error', "SYST:ERR", "%s", type=types.StringType, flags=Instrument.FLAG_GET) #good
        self.add_scpi_parameter('active_chan', "SYST:ACT:CHAN", '%s', type=types.StringType, flags=Instrument.FLAG_GET)
        self.add_scpi_parameter('active_measurement', "SYST:ACT:MEAS", '%s', type=types.StringType, flags=Instrument.FLAG_GET) #new

#        for lab in [['mixer',1],['VNA',2],['SPEC',3]]:            
#             self.add_scpi_parameter(lab[0]+'_INP_freq_fixed', 'SENS'+lab[1]':MIX:INP:FREQ:FIX', '%d', units='Hz', type = types.FloatType, gui_group=lab[0])
            
        self.add_scpi_parameter('sweep_time', 'SENS'+ch+':SWE:TIME', '%.8f', units="s", type=types.FloatType,
                                flags=Instrument.FLAG_GET, gui_group='sweep') # now good, yay
        self.add_scpi_parameter('segment_sweep_time', 'SENS'+ch+':SEGM:SWE:TIME', '%.8f', units="s", type=types.FloatType,
                                flags=Instrument.FLAG_GET, gui_group='sweep') #now good
        
        # mixer parameters for SMC (or any frequency convertring measurement class)
        if meas_class == 'SMC':
            # Scalar Mixer/Converter class parameters
            self.add_scpi_parameter('mixer_avoid_spurs', 'SENS'+ch+':MIX:AVO', '%i', type=bool, flags=Instrument.FLAG_GETSET, gui_group='mixer')
            self.add_scpi_parameter('mixerLO_freq_fixed', 'SENS'+ch+':MIX:LO:FREQ:FIX', "%d", units="Hz", type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerLO_freq_start', 'SENS'+ch+':MIX:LO:FREQ:STAR', "%d", units="Hz", type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerLO_freq_stop', 'SENS'+ch+':MIX:LO:FREQ:STOP', "%d", units="Hz", type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerOUTP_freq_fixed', 'SENS'+ch+':MIX:OUTP:FREQ:FIX', '%d', units='Hz', type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerOUTP_freq_start', 'SENS'+ch+':MIX:OUTP:FREQ:STAR', '%d', units='Hz', type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerOUTP_freq_stop', 'SENS'+ch+':MIX:OUTP:FREQ:STOP', '%d', units='Hz', type=types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerINP_freq_fixed', 'SENS'+ch+':MIX:INP:FREQ:FIX', '%d', units='Hz', type = types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerINP_freq_start', 'SENS'+ch+':MIX:INP:FREQ:STAR', '%d', units='Hz', type = types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerINP_freq_stop', 'SENS'+ch+':MIX:INP:FREQ:STOP', '%d', units='Hz', type = types.FloatType, gui_group='mixer')
            self.add_scpi_parameter('mixerINP_freq_mode', 'SENS'+ch+':MIX:INP:FREQ:MODE', '%s',
                                    type=types.StringType, flags=Instrument.FLAG_GETSET,
                                    format_map={'FIXED': 'FIXED', 'SWEPT': 'SWEPT'}, gui_group='mixer')
            self.add_scpi_parameter('mixerOUTP_freq_mode', 'SENS'+ch+':MIX:OUTP:FREQ:MODE', '%s',
                                    type=types.StringType, flags=Instrument.FLAG_GETSET,
                                    format_map={'FIXED': 'FIXED', 'SWEPT': 'SWEPT'}, gui_group='mixer')
            self.add_scpi_parameter('mixerLO_freq_mode', 'SENS'+ch+':MIX:LO:FREQ:MODE', '%s',
                                    type = types.StringType, flags=Instrument.FLAG_GETSET,
                                    format_map={'FIXED': 'FIXED', 'SWEPT': 'SWEPT'}, gui_group='mixer')
            self.add_scpi_parameter('mixer_xaxis', 'SENS'+ch+':MIX:XAX', '%s',
                                    type = types.StringType, flags=Instrument.FLAG_GETSET, gui_group='mixer',
                                    format_map={'INPUT': 'input', 'LO_1': 'LO_1', 'LO_2': 'LO_2', 'OUTPUT': 'output'})
            self.add_function('mixer_apply_settings')
            self.add_function('mixer_discard_settings')
            
            self.add_scpi_parameter('aux_trigger_1_out', 'TRIG:CHAN'+ch+':AUX1', '%i',
                               type=bool, flags=Instrument.FLAG_GETSET)                                       
    
            self.add_scpi_parameter('aux_trigger_1_out_interval', 'TRIG:CHAN'+ch+':AUX1:INT', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GETSET,
                                format_map = {"POIN": "point",
                                              "SWE": "sweep"})    
            
            
        elif meas_class == 'SA':
            # spectrum analyzer specific parameters
            self.add_scpi_parameter('spec_rbw_shape', 'SENS'+ch+':SA:BAND:SHAP', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='spec',
                                    format_map={'GAUS':'Gaussian', 'FLAT':'flat top', 'KAIS':'Kaiser',
                                                'BLAC':'Blackman', 'NONE':'none'})
            self.add_scpi_parameter('spec_detector_func', 'SENS'+ch+':SA:DET:FUNC', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='spec',
                                    format_map={'PEAK':'peak', 'AVER':'average', 'SAMP':'sample',
                                                'NORM':'normal', 'PSAM':'peak sample', 'PAV':'peak average'})
            self.add_scpi_parameter('spec_vbw_aver_type', 'SENS'+ch+':SA:BAND:VID:AVER:TYPE', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='spec',
                                    format_map={'VOLT':'voltage', 'POW':'power', 'LOG':'log',
                                                'VMAX':'voltage max', 'VMIN':'voltage min'})
            self.add_scpi_parameter('spec_rbw', 'SENS'+ch+':SA:BAND', '%d', units='Hz',
                                    type=types.FloatType, gui_group='spec')
            self.add_scpi_parameter('spec_vbw', 'SENS'+ch+':SA:BAND:VID', '%d', units='Hz',
                                    type=types.FloatType, gui_group='spec')
        elif meas_class == 'NFCS':
            self.add_scpi_parameter('noise_average_factor', "SENS"+ch+":NOIS:AVER:COUN", "%d", type=types.IntType, gui_group='averaging') #good
            self.add_scpi_parameter('noise_averaging_state', 'SENS'+ch+':NOIS:AVER:STAT', '%i', type=bool, flags=Instrument.FLAG_GETSET, gui_group='averaging') #good          
            self.add_scpi_parameter('receiver_bandwidth', "SENS"+ch+":NOIS:BWID", "%d", units="Hz", type=types.FloatType, gui_group='averaging') #good
        elif meas_class == 'Standard':
            self.add_scpi_parameter('measurement', 'CALC'+ch+':PAR:MOD:EXT', '%s',
                                    type=types.StringType, flags=Instrument.FLAG_SET,
                                    format_map={"S11": "S11", "S12": "S12", "S13": "S13", "S14": "S14",
                                                "S21": "S21", "S22": "S22", "S23": "S23", "S24": "S24",
                                                "S31": "S31", "S32": "S32", "S33": "S33", "S34": "S34",
                                                "S41": "S41", "S42": "S42", "S43": "S43", "S44": "S44"}) # now good, changed to only a set, doesn't actually work
        elif meas_class == 'SweptIMD':
            # swept IMD specific parameters
            self.add_scpi_parameter('imd_sweep_type', 'SENS'+ch+':IMD:SWE:TYPE', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='sweptIMD',
                                    format_map={'FCEN': 'center frequency',
                                                'DFR': 'delta frequency',
                                                'POW': 'power',
                                                'CW': 'cw mode, all constant',
                                                'SEGM': 'segment sweep of center frequency',
                                                'LOP': 'LO power'})
            self.add_scpi_parameter('imd_IFBW_main', 'SENS'+ch+':IMD:IFBW:MAIN', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_IFBW_im', 'SENS'+ch+':IMD:IFBW:IMT', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_tpow_coupled', 'SENS'+ch+':IMD:TPOW:COUP', '%i', type=bool, gui_group='sweptIMD')
            
            # for IMD POW sweep
            self.add_scpi_parameter('imd_f1_cw', 'SENS'+ch+':IMD:FREQ:F1:CW', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f2_cw', 'SENS'+ch+':IMD:FREQ:F2:CW', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_fc_cw', 'SENS'+ch+':IMD:FREQ:FCEN:CW', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_df_cw', 'SENS'+ch+':IMD:FREQ:DFR:CW', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f1_pow_start', 'SENS'+ch+':IMD:TPOW:F1:STAR', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f1_pow_stop', 'SENS'+ch+':IMD:TPOW:F1:STOP', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f2_pow_start', 'SENS'+ch+':IMD:TPOW:F2:STAR', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f2_pow_stop', 'SENS'+ch+':IMD:TPOW:F2:STOP', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')    
            # for IMD FCEN sweep
            self.add_scpi_parameter('imd_fc_start', 'SENS'+ch+':IMD:FREQ:FCEN:STAR', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_fc_stop', 'SENS'+ch+':IMD:FREQ:FCEN:STOP', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_fc_center', 'SENS'+ch+':IMD:FREQ:FCEN:CENT', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_fc_span', 'SENS'+ch+':IMD:FREQ:FCEN:SPAN', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f1_pow', 'SENS'+ch+':IMD:TPOW:F1', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_f2_pow', 'SENS'+ch+':IMD:TPOW:F2', '%.2f', units='dBm', type=types.FloatType, gui_group='sweptIMD')
            # for IMD DFR sweep
            self.add_scpi_parameter('imd_df_start', 'SENS'+ch+':IMD:FREQ:DFR:STAR', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
            self.add_scpi_parameter('imd_df_stop', 'SENS'+ch+':IMD:FREQ:DFR:STOP', '%d', units='Hz', type=types.FloatType, gui_group='sweptIMD')
        elif meas_class == 'IMSpec':
            self.add_scpi_parameter('ims_rbw', 'SENS'+ch+':IMS:RBW', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_resp_start', 'SENS'+ch+':IMS:RESP:STAR', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_resp_stop', 'SENS'+ch+':IMS:RESP:STOP', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_resp_center', 'SENS'+ch+':IMS:RESP:CENT', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_resp_span', 'SENS'+ch+':IMS:RESP:SPAN', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_df', 'SENS'+ch+':IMS:STIM:DFR', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_fc', 'SENS'+ch+':IMS:STIM:FCEN', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_f1', 'SENS'+ch+':IMS:STIM:F1FR', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_f2', 'SENS'+ch+':IMS:STIM:F2FR', '%d', units='Hz', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_sweep_type', 'SENS'+ch+':IMS:SWE:TYPE', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='IMSpec',
                                    format_map={'LIN': 'LINear',
                                                'SEC': 'SECond order products',
                                                'THIR': 'THIRd order products',
                                                'NTH': 'NTH order products'})
            self.add_scpi_parameter('ims_sweep_order', 'SENS'+ch+':IMS:SWE:ORD', '%d', type=types.IntType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_tpow_coupled', 'SENS'+ch+':IMS:TPOW:COUP', '%i', type=bool, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_f1_pow', 'SENS'+ch+':IMS:STIM:TPOW:F1', '%.2f', units='dBm', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_stim_f2_pow', 'SENS'+ch+':IMS:STIM:TPOW:F2', '%.2f', units='dBm', type=types.FloatType, gui_group='IMSpec')
            self.add_scpi_parameter('ims_tpow_level_mode', 'SENS'+ch+':IMS:TPOW:LEV', '%s', type=types.StringType,
                                    flags=Instrument.FLAG_GETSET, gui_group='IMSpec',
                                    format_map={'NONE': 'NONE - (Set Input Power)',
                                                'INPUT': 'INPUT - (Set Input Power, receiver leveling)',
                                                'EQUAL': 'EQUAL - (Set Input Power, equal tones at output)',
                                                'OUTPUT': 'OUTPUT - (Set Output Power, receiver leveling)'})
            
        
        #TODO - implement trace dependent electrical delay and offsets
        self.add_scpi_parameter('electrical_delay', "CALC"+ch+":CORR:EDEL:TIME", '%.8f', units="s", type=types.FloatType) #now good
        self.add_scpi_parameter('phase_offset', "CALC"+ch+":OFFS:PHAS", "%.4e", units="deg", type=types.FloatType) #added
        
        self.add_scpi_parameter('format', 'CALC'+ch+':FORM', '%s',
                                type=types.StringType, flags=Instrument.FLAG_GETSET,
                                format_map={"MLOG": "log mag",
                                       "PHAS": "phase",
                                       "GDEL": "group delay",
                                       "SMIT": "Smith R+jX",
                                       "SADM": "Smith G+jB",
                                       "POL": "polar",
                                       "MLIN": "linear mag",
                                       "SWR": "VSWR",
                                       "REAL": "real",
                                       "IMAG": "imaginary",
                                       "UPH": "unwrapped phase",
                                       "KELV": "Kelvin",
                                       "FAHR": "Farenheit",
                                       "CELS": "Celsius"}) #now good, no more SCOM...
        # triggering
        self.add_scpi_parameter('trigger_source', 'TRIG:SOUR', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GETSET,
                           format_map={"IMM": "immediate internal",
                                       "EXT": "external",
                                       "MAN": "manual"})                          
                                       
        self.add_scpi_parameter('trigger_mode', 'SENS'+ch+':SWE:MODE', '%s',
                                type=types.StringType, flags=Instrument.FLAG_GETSET,
                                format_map = {"HOLD": "HOLD channel",
                                              "CONT": "CONTinuous",
                                              "GRO": "GROups",
                                              "SING": "SINGle"})
        self.add_scpi_parameter('trigger_group_count', 'SENS'+ch+':SWE:GRO:COUN', "%d", type=types.IntType)
#        self.add_scpi_parameter('instrument_state_data', 'MMEM:STOR:STYP', '%s',
#                           type=types.StringType, flags=Instrument.FLAG_GETSET,
#                           format_map={"STAT": "measurement conditions only",
#                                       "CST": "+ calibration",
#                                       "DST": "+ data",
#                                       "CDST": "+ calibration + data"})
        if meas_class != 'SweptIMD':
            self.add_scpi_parameter('sweep_type', 'SENS'+ch+':SWE:TYPE', '%s', gui_group='sweep',
                                    type=types.StringType, flags=Instrument.FLAG_GETSET,
                                   format_map={"LIN": "linear",
                                               "LOG": "logarithmic",
                                               "SEGM": "segment",
                                               "POW": "power",
                                               "PHAS": "phase",
                                               "CW": "CW time"}) #now good
        
        self.add_scpi_parameter('smoothing', 'CALC'+ch+':SMO', '%i', type=bool,
                                flags=Instrument.FLAG_GETSET, gui_group='averaging')
#        self.add_scpi_parameter('arbitrary_segments', 'SENS:SEGM:ARB', '%s',
#                           type=types.StringType, flags=Instrument.FLAG_GETSET,
#                           format_map={"1": "allowed",
#                                       "0": "disallowed"}) # requires firmware upgrade
        self.add_scpi_parameter('clock_reference', 'SENS:ROSC:SOUR', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GET,
                           format_map={"INT": "internal",
                                       "EXT": "external"})
        self.add_parameter('instrument_state_file', type=types.StringType,
                           flags=Instrument.FLAG_GETSET)
#        self.add_parameter('fun1', type = types.Float, flags=Instrument.FLAG_GETSET) 
        self.add_function("save_state")
        self.add_function("load_state")
        self.add_function("autoscale")
        self.add_function("track_L")
        self.add_function("track_R")
        self.add_function("restart_averaging")
        self._instrument_state_file = None
        self.set(kwargs)
    
        
    def do_enable_averaging(self, enable=True):
        s = 'ON' if enable else 'OFF'
        self.set_averaging_state(s)
        self.restart_averaging() #clears data and restarts averaging
    
    def restart_averaging(self):
        ch = str(self._chan)
        self.write('SENS'+ch+':AVER:CLE') #clears data and restarts averaging

    # gets y data in format defined by fmt, returns 2D array with indices [real/imag, point_num], or 1D array [point_num]
    def do_get_data(self, fmt='POL', opc=False, trig_each_avg=False, trace_num=1):
        # fmt = 'POL' or any valid data format defined above
        # opc: if True, will only return data once averaging is complete. If False, returns current data.
        # trig_each_avg: if True, will trigger each individual average from the computer (this is slower and not implemented yet)
        #TODO - implement triggering each average if we want to        
        prev_fmt = self.get_format()
        if opc:
            prev_trig = self.get_trigger_source()
            prev_trig_mode = self.get_trigger_mode()
            prev_avg_state = self.get_averaging_state()
            prev_avg_factor = self.get_average_factor()
            self.set_trigger_source('IMM')
            avg_steps = prev_avg_factor
            if not bool(prev_avg_state): #if not averaging, force it to average with count 1
                avg_steps = 2
                self.set_average_factor(avg_steps)
                self.set_averaging_state(True)
#            self.set_trigger_group_count(avg_steps)
            self.set_trigger_mode('CONT') #no longer need GRO since we have averaging bit
            self.restart_averaging()
            # waits until a single trace is done averaging
            self.opc()
            sleep(100e-3)
            while (not self.averaging_done(trace_num)):
                sleep(100e-3) # wait 100 ms before asking if averaging is complete
            self.opc()
            if not bool(prev_avg_state):
                self.set_average_factor(prev_avg_factor)
                self.set_averaging_state(prev_avg_state)
        self.set_format(fmt)
        data = self.do_get_yaxes(fmt=fmt)
        self.set_format(prev_fmt)
        if opc:
            self.set_trigger_source(prev_trig)
            self.set_trigger_mode(prev_trig_mode)
        return data
    
    # returns True (actually an int (1<<bit_num) > 1) if trace_num is done averaging, 0/False if otherwise
    def averaging_done(self, trace_num):
        reg = ((int(trace_num) - 1) / 14) + 1 # register number to determine when trace_num trace is done averaging, ave_reg = 1 for traces 1-14
        bit_num = (int(trace_num) - 1) % 14 + 1  # corresponding bit number in that register, 1=done, 0=still averaging
        bit_weight = 1<<bit_num # mask for selecting bit_num bit
        val = int(self.ask('STAT:OPER:AVER' + str(reg) + ':COND?'))
        return val & bit_weight # selects out bit for relevant trace
    
    # gets swept axis
    def do_get_xaxis(self):
        ch = str(self._chan)
        return np.array(map(float, self.ask('CALC'+ch+':X?', timeout=0.1).split(',')))

    # gets data from currently selected measurement trace
    def do_get_yaxes(self, fmt='POL'):
        ch = str(self._chan)
        strdata = self.ask('CALC'+ch+':DATA? FDATA', timeout=1e3)
        data = np.array(map(float, strdata.split(',')))
        if fmt in ['POL', 'SMIT']:
            data = data.reshape((len(data)/2, 2))
            return data.transpose() # (mags, phases) or (real, imag)
        return data # only 1D array

    def reset(self):
        self.write('*RST')

    def opc(self):
        return self.ask('*OPC?')

    def trigger(self):
        self.write('INIT:IMM') #different

#TODO - check
    def do_multipoint_sweep(self, start, stop, step):
        n_points = self.get_points()
        span = n_points * step
        for start in np.arange(start, stop, span):
            self.set_start_freq(start)
            self.set_stop_freq(start + span)
            yield self.do_get_data()
#TODO - check
    def do_power_sweep(self, start, stop, step):
        for i, power in enumerate(np.arange(start, stop, step)):
            self.set_power(power)
            yield self.do_get_data()

    def do_get_chan(self):
        return self._chan
    
    def do_get_meas_class(self):
        return self._meas_class
        
    def do_set_instrument_state_file(self, filename):
        self._instrument_state_file = filename

    def do_get_instrument_state_file(self):
        return self._instrument_state_file

    def save_state(self, sfile=None):
        if sfile:
            self._instrument_state_file = sfile
        if self._instrument_state_file:
            self.write('MMEM:STOR "%s.sta"' % self._instrument_state_file) # maybe want " or "" around %s

    def load_state(self, sfile=None):
        if sfile:
            self._instrument_state_file = sfile
        if self._instrument_state_file:
            self.write('MMEM:LOAD "%s.sta"' % self._instrument_state_file)
        self.get_all()

    def autoscale(self, wind=1, tnum=1):
        w = str(wind)
        tr = str(tnum) #trace number within window (not global trace measurement number)
        self.write('DISP:WIND'+w+':TRAC'+tr+':Y:AUTO')

    def track_L(self):
        span = self.get_span()
        center = self.get_center_freq()
        self.set_center_freq(center-span/4)
        
    def track_R(self):
        span = self.get_span()
        center = self.get_center_freq()
        self.set_center_freq(center+span/4)

    # apply configured mixer settings for any freq converter application
    def mixer_apply_settings(self):
        ch = str(self._chan)
        self.write('SENS'+ch+':MIX:APPL')
    
    # calculates input, IF or output freqs of mixer setup
    def mixer_calc(self, param='OUTP'):
        # param: mixer port to be calculated (INPut, BOTH, OUTPut, LO_1, LO_2)
        ch = str(self._chan)
        self.write('SENS'+ch+':MIX:CALC ' + param)
    
    # cancels changes that have been made to the Converter setup and reverts to previous settings. Same as 'Cancel' button.
    def mixer_discard_settings(self):
        ch = str(self._chan)
        self.write('SENS'+ch+':MIX:DISC')
        
#    def load_segment_table(self, table, transpose=False):
#        # table should be of the form (array of starts, array of stops, array of points)
#        header = '6,0,0,0,0,0,0,%d,'%(np.size(table[0]))
#        starts = ['%.1f'%i for i in table[0]]
#        stops = ['%.1f'%i for i in table[1]]
#        points = ['%i'%i for i in table[2]]
#        if transpose:
#            indata = (starts,stops,points)
#        else:
#            indata = zip(*(starts,stops,points))
#        flat = [a for b in indata for a in b]
#        data = ','.join(flat)
#        self.write('SENS:SEGM:DATA %s%s' % (header,data))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    Instrument.test(Keysight_N5242A)
    # vna.do_get_data()
