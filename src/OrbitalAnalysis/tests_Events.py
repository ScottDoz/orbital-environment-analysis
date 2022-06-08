# -*- coding: utf-8 -*-
"""
Created on Fri May  6 21:03:49 2022

@author: scott

Test Events module

"""

#%% Events. Eclipses and Station Lighting

def test_station_lighting():
    
    # Select station
    # gs = 'DSS-43'
    gs = 'SSR-1'
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=False)
    start_et = et[0]
    stop_et = et[-1]
    # light, dark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-0.25)
    light, dark = find_station_lighting(start_et,stop_et,station=gs,ref_el=-6.)
    # light, dark = find_station_lighting(start_et,stop_et,station=gs,method='eclipse')
    
    
    result = dark
    
    # Print Dark Intervals
    print('Station Dark times')
    print('Start (UTC)                 Stop (UTC)                  Duration (s)')
    count = spice.wncard( result ) # Count of intervals
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    for i in range(count):
        beg,end = spice.wnfetd (result, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
    print('')
    
    
    # Print Sunlit Intervals
    result = light
    print('Station Light times')
    print('Start (UTC)                 Stop (UTC)                  Duration (s)')
    count = spice.wncard( result ) # Count of intervals
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    for i in range(count):
        beg,end = spice.wnfetd (result, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
    
    return dark, light

def test_sat_lighting():
    ''' Get the eclipse times of a satellite '''
    
    # Confirmed matches with output of GMAT EclipseLocator.
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=False)
    start_et = et[0]
    stop_et = et[-1]
    light, partial, dark = find_sat_lighting(start_et,stop_et)
    
    # Join full and partial results
    # resunion = spice.wnunid(light, partial)
    
    
    for lighting in ['Dark','Partial','Full']:
        
        # Select result
        if lighting == 'Full':
            result = light
        elif lighting == 'Partial':
            result = partial
        elif lighting == 'Dark':
            result = dark
        
        # Print results
        print('{} '.format(lighting))
        print('Start (UTC)                 Stop (UTC)                  Duration (s)')
        count = spice.wncard( result ) # Count of intervals
        # TIMFMT = "YYYY-MON-DD HR:MN:SC.###### (TDB) ::TDB ::RND" # Time format for printing
        TIMFMT = "DD-MON-YYYY HR:MN:SC.###### ::UTC ::RND" # Time format for printing
        
        TIMLEN = 41
        if count==0:
            print('No occultation was found.\n')
            continue
        
        for i in range(count):
            beg,end = spice.wnfetd (result, i)
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            
            # print( "Interval {}".format( i + 1));
            print( "{} {} {}".format( begstr, endstr, str(dur)) );
            # print( " \n" );
        print('')
    
    return result

def test_access():
    
    # TODO: Check receive vs transmit results
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_et = et[0]
    stop_et = et[-1]
    
    # Select groundstation
    # gs = 'DSS-43'
    gs = 'SSRD-1'
    # gs = 'SSR-1'
    # gs = 'SSR-2'    
    
    # Compute line-of-sight access intervals
    los_access = find_access(start_et,stop_et,station=gs)
    dflos = window_to_dataframe(los_access)
    
    # Compute station lighting intervals
    # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs, ref_el=-6.)
    gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs, ref_el=-0.25)
    # gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs, method='eclipse')
    
    # Compute satellite lighting intervals
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    satlight1 = spice.wnunid(satlight,satpartial) # Full or partial light
    
    # Compute visible (constrained) access intervals
    access = constrain_access_by_lighting(los_access,gsdark,satlight1)
    dfaccess = window_to_dataframe(access)
    
    # Print results
    
    # Line-of-sight access (no constraints)
    count = spice.wncard( los_access ) # Count of intervals
    print('Line-of-Sight Access')
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    print('# Start (UTC)                 Stop (UTC)                  Duration (s)')
    for i in range(count):
        beg,end = spice.wnfetd (los_access, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {} {}".format(i+1, begstr, endstr, str(dur)) );
    print('')
    print('Min Duration {} s'.format(dflos.Duration.min()))
    print('Max Duration {} s'.format(dflos.Duration.max()))
    print('Mean Duration {} s'.format(dflos.Duration.mean()))
    print('Total Duration {} s'.format(dflos.Duration.sum()))
    print('')
    print('')
    
    # Visible access (no constraints)
    count = spice.wncard( access ) # Count of intervals
    print('Visible Access')
    TIMFMT = "YYYY-MON-DD HR:MN:SC.###### ::UTC ::RND" # Time format for printing
    TIMLEN = 41
    print('# Start (UTC)                 Stop (UTC)                  Duration (s)')
    for i in range(count):
        beg,end = spice.wnfetd (access, i)
        if beg==end:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            print("Event time: {} \n".format(begstr))
        else:
            begstr = spice.timout ( beg, TIMFMT, TIMLEN)
            endstr = spice.timout ( end, TIMFMT, TIMLEN)
            dur = end-beg
            print( "{} {} {} {}".format(i+1, begstr, endstr, str(dur)) );
    print('')
    print('Min Duration {} s'.format(dfaccess.Duration.min()))
    print('Max Duration {} s'.format(dfaccess.Duration.max()))
    print('Mean Duration {} s'.format(dfaccess.Duration.mean()))
    print('Total Duration {} s'.format(dfaccess.Duration.sum()))
    print('')
    
    return access


#%% Plot Overpass

def test_plot_access():
    
    gs = 'DSS-43'
    # gs = 'SSR-30'
    
    # Generate ephemeris times
    step = 10.
    # step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    start_et = et[0]
    stop_et = et[-1]
    
    # Compute satellite lighting intervals
    satlight, satpartial, satdark = find_sat_lighting(start_et,stop_et)
    
    # Compute station lighting intervals
    gslight, gsdark = find_station_lighting(start_et,stop_et,station=gs)
    
    # Compute line-of-sight access intervals
    access = find_access(start_et,stop_et,station=gs)
    
    # Plot
    plot_access_times(access,gslight,gsdark,satlight, satpartial, satdark)
    
    
    return

def test_plot_overpass():
    
    # Load access
    dfa = load_access_results()
    
    # Generate ephemeris times
    # step = 10.
    step = 5.
    et = generate_et_vectors_from_GMAT_coverage(step, exclude_ends=True)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et)[0]
    # dftopo['Sat.El'] = np.rad2deg(dftopo['Sat.El'])
    # dftopo['Sat.Az'] = np.rad2deg(dftopo['Sat.Az'])
    
    # Plot
    # plot_overpass(dftopo, dfa)
    plot_overpass_magnitudes(dftopo, dfa)
    
    return

def test_plot_visual_magnitude():
    
    
    # Generate ephemeris times
    et = generate_et_vectors_from_GMAT_coverage(30., exclude_ends=True)
    
    # Get Topocentric observations
    dftopo = get_ephem_TOPO(et,groundstations=['SSR-1','SSR-2'])
    dftopo = dftopo[0] # Select first station
    
    
    # Compute Visual magnitudes
    Rsat = 1 # Radius of satellite (m)
    msat = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12) # With airmass
    msat2 = compute_visual_magnitude(dftopo,Rsat,p=0.25,k=0.12,include_airmass=False) # Without airmass
    
    # Generate plots
    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Visual Magnitude')
    # Magnitude vs elevation
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat,'.b')
    ax1.plot(np.rad2deg(dftopo['Sat.El']),msat2,'.k')
    ax1.set_xlabel("Elevation (deg)")
    ax1.set_ylabel("Visual Magnitude (mag)")
    ax1.invert_yaxis() # Invert y axis
    # Az/El
    ax2.plot(dftopo['ET'],msat,'-b')
    ax2.plot(dftopo['ET'],msat2,'-k')
    ax2.invert_yaxis() # Invert y axis
    ax2.set_xlabel("Epoch (ET)")
    ax2.set_ylabel("Visual Magnitude (mag)")
    fig.show()
    
    
    return
