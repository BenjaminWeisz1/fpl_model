"""Synthetic full-history browser discovery/import/restart tests; no network."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fpl_model.fetch import fetch_sofascore as f
from fpl_model.fetch import sofascore_browser_backfill as b
from fpl_model.fetch.player_identity import Registry
from fpl_model.fetch.test_sofascore import event, lineup


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.raw,self.data,self.run=self.root/'raw',self.root/'current',self.root/'run'
        self.run.mkdir(); self.data.mkdir()
        f.atomic_json(self.run/'run_context.json',{'started_at':f.now()})
        f.atomic_json(self.run/'navigation.json',[])
        self.client=f.Client(raw=self.raw)

    def tearDown(self): self.tmp.cleanup()

    def metadata(self,route,payload):
        self.client.store(b.metadata_path(self.raw,route),payload,f.BASE+route)

    def fixture(self,eid,season,sid):
        return dict(event(eid),season={'id':sid,'year':f'{season}/{season+1}'},
                    tournament={'uniqueTournament':{'id':17}},
                    homeTeam={'id':1,'name':'Arsenal'},awayTeam={'id':2,'name':'Liverpool'})

    def test_multiseason_pagination_unknown_discovery_and_restart(self):
        self.metadata('/unique-tournament/17/seasons',{'seasons':[{'id':1,'year':'2017/18'},{'id':2,'year':'2018/19'}]})
        for sid in (1,2): self.metadata(f'/unique-tournament/17/season/{sid}/rounds',{'rounds':[{'round':1}]})
        route='/unique-tournament/17/season/1/events/round/1'
        self.metadata(route,{'events':[self.fixture(10,2017,1)],'hasNextPage':True})
        result=b.plan(self.raw,self.data,self.run)
        self.assertFalse(result['discoveries'][2017]['verified'])
        self.assertFalse(result['discoveries'][2018]['verified'])
        self.assertFalse((self.raw/'discovery/2017.json').exists())
        self.metadata(route+'?page=1',{'events':[self.fixture(11,2017,1)],'hasNextPage':False})
        self.metadata('/unique-tournament/17/season/2/events/round/1',{'events':[self.fixture(20,2018,2)]})
        result=b.plan(self.raw,self.data,self.run)
        self.assertTrue(result['discoveries'][2017]['verified'])
        self.assertEqual(result['discoveries'][2017]['enumerated_completed'],2)
        self.assertTrue(result['discoveries'][2018]['verified'])
        self.client.store(self.raw/'matches/2017/10.json',lineup(),f.BASE+'/event/10/lineups',f.validate_lineup,
                          season=2017,event=self.fixture(10,2017,1))
        result=b.plan(self.raw,self.data,self.run)
        queue=json.loads((self.run/'queue.json').read_text())
        self.assertEqual({j['event']['id'] for j in queue if j['kind']=='lineup'},{11,20})
        before=(self.raw/'matches/2017/10.json').read_bytes()
        b.plan(self.raw,self.data,self.run)
        self.assertEqual(before,(self.raw/'matches/2017/10.json').read_bytes())

    def test_dom_sync_preserves_signed_fields_and_original_provenance(self):
        e=self.fixture(10,2017,1); route='/event/10/lineups'; dest=self.raw/'matches/2017/10.json'
        f.atomic_json(self.run/'jobs.json',[dict(kind='lineup',route=route,cache=str(dest),season=2017,season_id=1,event=e)])
        payload=lineup();payload['home']['players'][0]['statistics']['goalsPrevented']=-.8
        dom=self.run/'dom.json';f.atomic_json(dom,payload)
        obs=dict(route=route,url=f.BASE+route,outcome='parsed_json',file=str(dom),end=f.now())
        f.atomic_json(self.run/'navigation.json',[obs])
        self.assertEqual(b.sync(self.raw,self.run)['responses_imported'],1)
        before=dest.read_bytes()
        self.assertEqual(b.sync(self.raw,self.run)['responses_imported'],0)
        self.assertEqual(before,dest.read_bytes())
        env=self.client.load(dest,f.validate_lineup)
        self.assertEqual(env['payload'],payload);self.assertEqual(env['dom_sha256'],b.sha(dom))
        self.assertEqual(env['retrieved_at'],obs['end'])
        # A changed/truncated file must invalidate the cheap frozen-cache index.
        dest.write_text('{truncated',encoding='utf-8')
        with self.assertRaises(f.FetchError):b.sync(self.raw,self.run)
        self.assertFalse(dest.exists())
        self.assertTrue(list(dest.parent.glob('*.invalid.*')))

    def test_wrong_season_event_rejected_and_no_discovery_written(self):
        self.metadata('/unique-tournament/17/seasons',{'seasons':[{'id':1,'year':'2017/18'}]})
        self.metadata('/unique-tournament/17/season/1/rounds',{'rounds':[{'round':1}]})
        self.metadata('/unique-tournament/17/season/1/events/round/1',{'events':[self.fixture(10,2018,1)]})
        with self.assertRaises(ValueError): b.plan(self.raw,self.data,self.run)
        self.assertFalse((self.raw/'discovery/2017.json').exists())

    def test_unavailable_lineup_recorded_while_independent_fixture_imports(self):
        jobs=[]; observations=[]
        for eid,payload in [(10,{'home':{'players':[]}}),(11,lineup())]:
            route=f'/event/{eid}/lineups';dom=self.run/f'dom_{eid}.json'
            f.atomic_json(dom,payload)
            jobs.append(dict(kind='lineup',route=route,cache=str(self.raw/f'matches/2017/{eid}.json'),
                             season=2017,season_id=1,event=self.fixture(eid,2017,1)))
            observations.append(dict(route=route,url=f.BASE+route,outcome='parsed_json',file=str(dom),end=f.now()))
        f.atomic_json(self.run/'jobs.json',jobs);f.atomic_json(self.run/'navigation.json',observations)
        result=b.sync(self.raw,self.run)
        self.assertEqual(result['responses_imported'],1)
        self.assertEqual(result['validation_failures'],1)
        self.assertFalse((self.raw/'matches/2017/10.json').exists())
        self.assertTrue((self.raw/'matches/2017/11.json').exists())
        failure=json.loads((self.run/'validation_failures.json').read_text())
        self.assertEqual(failure['/event/10/lineups']['kind'],'unavailable')

    def test_cross_side_canonical_collision_does_not_publish(self):
        self.client.store(self.raw/'matches/2017/10.json',lineup(),f.BASE+'/event/10/lineups',f.validate_lineup,
                          season=2017,event=self.fixture(10,2017,1))
        registry=Registry(self.data)
        cid=registry.ensure('understat','7','Synthetic accepted player')
        registry.add_alias('sofascore','1',cid)
        registry.add_alias('sofascore','101',cid)
        with self.assertRaisesRegex(ValueError,'across source event sides'):
            f.build(self.client,registry,self.data)
        self.assertFalse((self.data/'sofascore_player_match_stats.csv').exists())

    def test_reviewed_fixture_spelling_preserves_both_accepted_ids(self):
        e=self.fixture(10,2017,1);e['awayTeam']['name']='Sheffield United'
        self.client.store(self.raw/'matches/2017/10.json',lineup(),f.BASE+'/event/10/lineups',f.validate_lineup,
                          season=2017,event=e)
        registry=Registry(self.data)
        source_id=registry.fixture('sofascore',10,2017,'Arsenal','Sheffield United','2017-08-12')
        local_id=registry.fixture('understat',20,2017,'Arsenal','Sheffield Utd','2017-08-12')
        registry.save()
        result=b.normalize_fixture_aliases(self.raw,self.data,self.run)
        self.assertEqual(result['reviewed_changes'],1)
        loaded=Registry(self.data)
        self.assertEqual(loaded.fixtures['sofascore','10']['Canonical_Fixture_ID'],source_id)
        self.assertEqual(loaded.fixtures['understat','20']['Canonical_Fixture_ID'],local_id)
        self.assertEqual(loaded.fixtures['sofascore','10']['Away'],'Sheffield Utd')
        self.assertTrue((self.run/'fixture_alias_review.json').exists())
        self.assertEqual(b.normalize_fixture_aliases(self.raw,self.data,self.run)['reviewed_changes'],0)

    def test_current_refresh_remains_pending_across_partial_batches(self):
        listing='/unique-tournament/17/seasons';rounds='/unique-tournament/17/season/1/rounds'
        events='/unique-tournament/17/season/1/events/round/1'
        payloads={listing:{'seasons':[{'id':1,'year':'2017/18'}]},rounds:{'rounds':[{'round':1}]},
                  events:{'events':[self.fixture(10,2017,1)]}}
        for route,payload in payloads.items():self.metadata(route,payload)
        def observe(route,payload):
            dom=self.run/(b.route_key(route)+'.dom.json');f.atomic_json(dom,payload)
            nav=json.loads((self.run/'navigation.json').read_text())
            nav.append(dict(route=route,url=f.BASE+route,outcome='parsed_json',file=str(dom),end=f.now()))
            f.atomic_json(self.run/'navigation.json',nav)
            b.sync(self.raw,self.run)
        with patch.object(f,'CURRENT_SEASON',2017):
            self.assertTrue(b.plan(self.raw,self.data,self.run)['discoveries'][2017]['verified'])
            b.plan(self.raw,self.data,self.run,refresh_current=True)
            observe(listing,payloads[listing])
            result=b.plan(self.raw,self.data,self.run)
            self.assertFalse(result['discoveries'][2017]['verified'])
            queue=json.loads((self.run/'queue.json').read_text())
            self.assertTrue(any(j['route']==rounds and j['refresh'] for j in queue))
            observe(rounds,payloads[rounds]);b.plan(self.raw,self.data,self.run)
            observe(events,{'events':[self.fixture(10,2017,1),self.fixture(11,2017,1)]})
            result=b.plan(self.raw,self.data,self.run)
            self.assertTrue(result['discoveries'][2017]['verified'])
            self.assertEqual(result['discoveries'][2017]['enumerated_completed'],2)
            incomplete=b.checkpoint(self.raw,self.data,self.run)
            self.assertFalse(incomplete['collection_complete'])
            self.assertEqual(incomplete['seasons']['2017']['missing'],2)
            for eid in (10,11):
                self.client.store(self.raw/f'matches/2017/{eid}.json',lineup(),
                                  f.BASE+f'/event/{eid}/lineups',f.validate_lineup,
                                  season=2017,event=self.fixture(eid,2017,1))
            complete=b.checkpoint(self.raw,self.data,self.run)
            self.assertTrue(complete['complete'])
            self.assertTrue(complete['collection_complete'])
            self.assertEqual(complete['pending'],{})

    def test_audit_rejects_cache_growth_during_validation(self):
        dom=self.run/'original.dom.json';f.atomic_json(dom,lineup())
        match=self.raw/'matches/2017/10.json'
        self.client.store(match,lineup(),f.BASE+'/event/10/lineups',f.validate_lineup,
                          season=2017,event=self.fixture(10,2017,1),dom_path=str(dom),dom_sha256=b.sha(dom))
        f.build(self.client,Registry(self.data),self.data)
        self.client.store(self.raw/'discovery/2017.json',{'events':[self.fixture(10,2017,1)]},'synthetic')
        f.atomic_json(self.run/'discovery_status.json',{'2017':{'verified':True}})
        original=f.Client.load;injected=[]
        def concurrent_import(client,path,validator=None):
            result=original(client,path,validator)
            if path==match and not injected:
                injected.append(True)
                self.client.store(self.raw/'matches/2017/11.json',lineup(),f.BASE+'/event/11/lineups',
                                  f.validate_lineup,season=2017,event=self.fixture(11,2017,1),
                                  dom_path=str(dom),dom_sha256=b.sha(dom))
            return result
        with patch.object(f,'DATA',self.data),patch.object(f,'ROOT',self.root),patch.object(f,'CURRENT_SEASON',2017):
            with patch.object(f.Client,'load',concurrent_import):
                with self.assertRaisesRegex(ValueError,'Raw cache changed during audit'):
                    b.audit(self.raw,self.data,self.run)
        self.assertFalse((self.run/'data_audit.json').exists())


if __name__=='__main__': unittest.main()
