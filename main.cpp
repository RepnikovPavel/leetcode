#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <bits/stdc++.h> 
using namespace std;


//  Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> h;
        std::vector<int> o_;
        auto N = nums.size();
        for(int i =0; i< N; i++){
            h.insert({nums[i],i});
        }
        for(int i=0;i<N;i++){
            auto ptr = h.find(target-nums[i]);
            if(ptr != h.end() && h[target-nums[i]]!=i){
                o_.push_back(i);
                o_.push_back(h[target-nums[i]]);
                break;
            }
        }
        return o_;
    }
    int romanToInt(string s) {
        int N = s.size();
        auto h = std::unordered_map<int,int>{
            {'I',1},
            {'V',5},
            {'X',10},
            {'L',50},
            {'C',100},
            {'D',500},
            {'M',1000}
        };
        auto mulh = std::unordered_map<std::string, int>{
            {"IV", 4},
            {"IX", 9},
            {"XL", 40},
            {"XC", 90},
            {"CD",400},
            {"CM",900}
        };

        int sum_ = 0;
        int i = N-1;
        while(i>-1){
            if(i==0){
                sum_ += h[s[i]];
                break;
            }
            auto sub_ = s.substr(i-1,2);
            if (mulh.find(sub_) != mulh.end())
            {
                sum_ += mulh[sub_];
                i -= 2;
            }
            else {
                // sum_ += h[sub_[0]];
                sum_ += h[sub_[1]];
                i -=1;
            }
        }            
        return sum_;
    }
    bool hasCycle(ListNode *head) {
        if(head == nullptr){
            return false;
        }
        auto h = std::unordered_map<ListNode*, int>();
        auto left = head/* condition */->next;
        ListNode* cv = head;
        int pos = 0;
        while(left != nullptr){
            if(h.find(cv) != h.end()){
                return true;
            }
            else{
                h.insert({cv, pos});
            }
            cv = left->next;
            left = left->next;
            pos += 1;
        }
        return false;
    }

    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB){
        if (headA==nullptr && headB != nullptr)
        {
            return headB;
        }
        if (headB==nullptr && headA != nullptr)
        {
            return headA;
        }
        if (headA==nullptr && headB==nullptr)
        {
            return nullptr;
        }
        auto ha = std::unordered_set<ListNode*>();
        auto hb = std::unordered_set<ListNode*>();
        auto cpa = headA;
        auto cpb = headB;
        int i = 1;
        while(true){
            if (cpa == nullptr && cpb == nullptr)
            {
                break;
            }
            if(i%2 ==0){
                if (cpa == nullptr && cpb != nullptr)
                {
                    i +=1;
                    continue;
                }
                if (hb.find(cpa) != hb.end())
                {
                    return cpa;
                }
                ha.insert(cpa);
                cpa = cpa->next;
            }
            if(i%2 != 0){
                if (cpb==nullptr && cpa != nullptr)
                {
                    i+=1;
                    continue;
                }
                if (ha.find(cpb) != ha.end())
                {
                    return cpb;
                }
                
                hb.insert(cpb);
                cpb = cpb->next;
            }
            i+=1;
        }
        return nullptr;
    }
    int majorityElement(vector<int>& nums) {
        int N = nums.size();
        auto h = std::unordered_map<int,int>();
        for (size_t i = 0; i < N; i++)
        {
            h.insert({nums[i],0});
        }
        for (size_t i = 0; i < N; i++)
        {
            h[nums[i]] += 1;
        }
        for(auto it: h){
            if(it.second > N/2){
                return it.first;
            }
        }

        
    }

    bool isHappy(int n) {
        // get digits
        // if sum(square(digits)) = 1 then true
        // retry
        int buff[10];
        int powers[10];
        int N =10; 
        for(int i=0;i<N;i++){
            powers[i]=(int)pow((float)10,(float)(N-i-1));
        }
        int tmp = n;
        int maxiter = 1000;
        while(maxiter--){
            for(int i=0;i<N;i++){
                buff[i] = tmp/powers[i];
                tmp = tmp - buff[i]*powers[i];
            }
            for(int i=0;i<N;i++){
                tmp += buff[i]*buff[i];
            }
            if (tmp == 1){
                return true;
            }
        }

        return false;
    }
    bool isIsomorphic(string s, string t) {
        auto hs = std::unordered_map<char, int>();
        auto ht = std::unordered_map<char, int>();
        int Ns = s.size();
        for(int i=0;i<Ns;i++){
            if (hs.find(s[i])==hs.end() && ht.find(t[i]) == ht.end())
            {
                hs[s[i]] = i;
                ht[t[i]] = i;
                continue;

            }
            else if (hs.find(s[i]) == hs.end() && ht.find(t[i]) != ht.end()){
                return false;
            }
            else if (hs.find(s[i]) != hs.end() && ht.find(t[i]) == ht.end()){
                return false;
            }
            else{
                if(hs[s[i]]!=ht[t[i]]){
                    return false;
                }
            }
            hs[s[i]] = i;
            ht[t[i]] = i;
        }

        return true;
    }
    bool containsDuplicate(vector<int>& nums) {
        auto h =  std::unordered_map<int,int>();
        int N = nums.size();
        for(int i=0;i<N;i++){
            h.insert({nums[i],0});
        }
        for(int i=0;i<N;i++){
            h[nums[i]] +=1;
        }
        for(auto el: h){
            if(el.second>1){
                return true;
            }
        }
        return false;
    }

    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        //Given an integer array nums and an integer k, 
        //return true if there are two distinct indices i and j 
        //in the array such that nums[i] == nums[j] and abs(i - j) <= k.
        auto h = std::unordered_map<int, std::vector<int>>();
        int N = nums.size();
        for(int i=0;i<N;i++){
            h.insert({nums[i], std::vector<int>()});
        }
        for(int i=0;i<N;i++){
            h[nums[i]].push_back(i);
        }
        for(auto& el: h){
            for(int i=0;i<el.second.size()-1;i++){
                if(abs(el.second[i]-el.second[i+1]) <= k){
                    return true;
                }
            }
        }
        return false;
    }
    bool isAnagram(string s, string t) {
        auto hs = std::unordered_map<char,int>();
        auto ht = std::unordered_map<char,int>();
        int Ns = s.size();
        int Nt = t.size();
        if(Ns != Nt){return false;}
        for(int i=0;i<Ns;i++){
            if(hs.find(s[i]) != hs.end()){
                hs[s[i]] +=1;
            }
            else{
                hs.insert({s[i],1});
            }

            if(ht.find(t[i]) != ht.end()){
                ht[t[i]] +=1;
            }
            else{
                ht.insert({t[i],1});
            }
        }
        for(int i=0;i<Ns;i++){
            if(ht.find(s[i]) != ht.end() && hs.find(t[i])==hs.end()){
                return false;
            }
            else if(ht.find(s[i])==ht.end() && hs.find(t[i]) != hs.end()){
                return false;
            }
            else if(hs[s[i]] != ht[s[i]]){
                return false;
            }
        }
        return true;
    }
    int missingNumber(vector<int>& nums) {
        // Given an array nums containing n distinct numbers in the range [0, n], 
        // return the only number in the range that is missing from the array.
        int N = nums.size();
        auto counts = std::vector<int>(N+1,0);
        for(int i=0;i<N;i++){
            counts[nums[i]] +=1;
        }
        return std::find(counts.begin(),counts.end(), 0)- counts.begin();
    }
    bool wordPattern(string pattern, string s) {
        int Np = pattern.size();
        int Ns = s.size();
        auto h = std::unordered_map<char, std::string>();
        auto word_pattern = std::unordered_map<std::string, char>();
        int start=0;
        int stop =-1;
        int pattern_pos = 0;
        for(int i=0;i<Ns;i++){
            if(i==Ns-1){
                if(pattern_pos != Np-1){
                    return false;
                }
                auto word = s.substr(start, stop-start+2);
                if(h.find(pattern[pattern_pos])==h.end()){
                    h.insert({pattern[pattern_pos],word});
                    if(word_pattern.find(word) == word_pattern.end()){
                        word_pattern.insert({word, pattern[pattern_pos]});
                    }
                    else if(word_pattern[word] != pattern[pattern_pos]){
                        return false;
                    }
                }
                else if (h[pattern[pattern_pos]] != word){
                    return false;
                }
                start = i+1;
                stop = i;
                pattern_pos += 1;
            }
            if(s[i] == ' '){
                auto word = s.substr(start, stop-start+1);
                if(h.find(pattern[pattern_pos])==h.end()){
                    h.insert({pattern[pattern_pos],word});
                    if(word_pattern.find(word) == word_pattern.end()){
                        word_pattern.insert({word, pattern[pattern_pos]});
                    }
                    else if(word_pattern[word] != pattern[pattern_pos]){
                        return false;
                    }
                }
                else if (h[pattern[pattern_pos]] != word){
                    return false;
                }
                start = i+1;
                stop = i;
                pattern_pos += 1;
            }
            else{stop += 1;}
        }
        return true;
    }
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        auto h1 = std::unordered_set<int>();
        auto h2 = std::unordered_set<int>();
        int N1 = nums1.size();
        int N2 = nums2.size();
        for(int i=0;i<N1;i++){
            h1.insert(nums1[i]);
        }
        for(int i=0;i<N2;i++){
            h2.insert(nums2[i]);
        }
        auto cap = std::vector<int>();
        cap.reserve(N1>N2?N1:N2);
        for(auto el:h1){
            if(h2.find(el) != h2.end()){
                cap.push_back(el);
            }
        }
        return cap;
    }
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        auto h1 = std::unordered_map<int,int>();
        auto h2 = std::unordered_map<int,int>();
        int N1 = nums1.size();
        int N2 = nums2.size();
        for(int i=0;i<N1;i++){
            h1[nums1[i]] += 1;
        }
        for(int i=0;i<N2;i++){
            h2[nums2[i]] += 1;
        }
        auto cap = std::vector<int>();
        cap.reserve(N1>N2?N1:N2);
        for(auto el:h1){
            if(h2.find(el.first) != h2.end()){
                for(int i=0;i<(el.second<h2[el.first]?el.second:h2[el.first]);i++){
                    cap.push_back(el.first);
                }
            }
        }
        return cap;
    }
    int distributeCandies(vector<int>& candyType) {
        // Input: candyType = [1,1,2,2,3,3]
        // Output: 3
        auto h = std::unordered_map<int,int>();
        int N = candyType.size();
        for(int i=0;i<N;i++){
            h[candyType[i]] += 1;
        }
        int sum_ = N/2;
        for(auto& el:h){
            if(sum_==0){
                break;
            }
            while(el.second > 1 && sum_ > 0){
                el.second--;
                sum_--;
            }
        }
        if(sum_!=0){
            for(auto& el:h){
                if(sum_==0){
                    break;
                }
                while(el.second >0 && sum_ > 0){
                    el.second--;
                    sum_--;
                }
            }
        }
        int cnt=0;
        for(auto& el:h){
            if(el.second > 0){
                cnt++;
            }
        }
        return cnt;

    }
    vector<int> fairCandySwap(vector<int>& aliceSizes, vector<int>& bobSizes) {
        int c1 = std::accumulate(aliceSizes.begin(),aliceSizes.end(),0);
        int c2 = std::accumulate(bobSizes.begin(),bobSizes.end(),0);
        int amt = (c1+c2)/2;
        int shortage1 = amt-c1;
        int shortage2 = amt-c2;
        auto s1 = std::unordered_set<int>(aliceSizes.begin(),aliceSizes.end());
        auto s2 = std::unordered_set<int>(bobSizes.begin(),bobSizes.end());
        for(auto a_i: s1){
            // int required_b = (2*a_i+shortage2-shortage1)/2;
            int required_b = (2*a_i+c2-c1)/2;
            if(s2.find(required_b)!=s2.end()){
                return std::vector<int>{a_i,amt+a_i-c1};
            }
        }
        return std::vector<int>{-1,-1};
    }
    int missingInteger(vector<int>& nums) {
        // find longest sequential prefix i=[0, N-1]
        // find smallest missing x: x>=sum(longest seq prefix)
        int N = nums.size();
        int longest_size = 1;
        for(int i=1;i<N;i++){
            if(nums[i]==nums[i-1]+1){
                longest_size +=1;
            }
            else{
                break;
            }
        }
        auto h = std::unordered_set<int>();
        for(int i=0;i<N;i++){
            h.insert(nums[i]);
        }
        int sum_ = std::accumulate(nums.begin(), nums.begin()+longest_size,0);
        int max_ = nums[std::max_element(nums.begin(), nums.end())-nums.begin()];
        if (sum_ > max_){
            return sum_;
        }
        for(int x=sum_; x<=max_;x++){
            if(h.find(x) == h.end()){
                return x;
            }
        }
        return max_+1;
    }
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        // Input: strs = ["eat","tea","tan","ate","nat","bat"]
        // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
        // check is anagram for each pair of words, y know O(N^2*mean_size_of_str)
        //1 <= strs.length <= 104
        //0 <= strs[i].length <= 100
        //strs[i] consists of lowercase English letters.
        
        auto anagrams = std::vector<std::vector<std::string>>(); 
        auto already_seen_positions = std::unordered_set<int>();
        int group_index = 0;
        for(int word_index =0; word_index < strs.size();word_index++){
            if(already_seen_positions.find(word_index) != already_seen_positions.end()){
                continue;
            }
            anagrams.push_back(std::vector<std::string>());
            anagrams[group_index].push_back(strs[word_index]);
            already_seen_positions.insert(word_index);
            for(int another_word_index = 0;another_word_index<strs.size();another_word_index++){
                if(already_seen_positions.find(another_word_index) != already_seen_positions.end()){
                    continue;
                }
                if(word_index==another_word_index){
                    continue;
                }
                if(isAnagram(strs[word_index],strs[another_word_index])){
                    anagrams[group_index].push_back(strs[another_word_index]);
                    already_seen_positions.insert(another_word_index);
                }
            }
            group_index++;
        }
        return anagrams;
    vector<int> topKFrequent(vector<int>& nums, int k) {
        int N = nums.size();
        auto h = std::unordered_map<int,int>();
        for(auto el: nums){
            h[el]++;
        }
        auto hcnt = std::unordered_map<int,std::vector<int>>{};
        for(auto& el: h){
            auto number = el.first;
            auto cnt = el.second;
            hcnt[cnt].push_back(number);
        }
        std::vector<int> freq(N+1,0);
        for(auto& el: hcnt){
            freq[N-el.first] = 1;
        }
        k = k>h.size()?h.size():k;
        std::vector<int> top_k_freq(k,0);
        int tmp_ = 0;
        for(int i =0 ;i < freq.size();i++){
            if(freq[i] == 1){
                top_k_freq[tmp_] = N-i;
                tmp_++;
            }
            if(tmp_>k-1){
                break;
            }
        }
        std::vector<int> top_k(k,0);
        int cp_ = 0;
        for(int freq_pos_=0;freq_pos_<k;freq_pos_++)
        {
            for(int i=0;i<hcnt[top_k_freq[freq_pos_]].size();i++){
                top_k[cp_] = hcnt[top_k_freq[freq_pos_]][i];
                cp_++;
                if(cp_ > k-1){
                    return top_k;
                }
            }
        }
        return top_k;
    }
    vector<int> productExceptSelf(vector<int>& nums) {
        //Input: nums = [1,2,3,4]
        //Output: [24,12,8,6]
        int N = nums.size();
        auto ans = std::vector<int>();
        ans.resize(N);
        int number_of_zeros = 0;
        int contain_zero =0;
        int P_ = 1;
        for(int i=0;i<N;i++){
            if(nums[i]==0){
                contain_zero = 1;
                number_of_zeros++;
            }
            else{
                P_ *= nums[i];
            }
        }
        for(int i=0;i<N;i++){
            if(nums[i] == 0){
                if(number_of_zeros > 1){
                    ans[i]=0;
                }
                else if(number_of_zeros==1){
                    ans[i] = P_;
                }
            }
            else{
                if(contain_zero){
                    ans[i]=0;
                }
                else{
                    ans[i] = P_/nums[i];
                }
            }
        }
        return ans;
    }


};


int main(){
    auto sol = Solution();
    auto v = std::vector<std::string>{"eat","tea","tan","ate","nat","bat"};
    auto ans = sol.groupAnagrams(v);
    // auto v1 = std::vector<int>{1,1};
    // auto v2 = std::vector<int>{2,2};
    // auto v2 = std::vector<int>{2,2};
    // auto v = std::vector<int>{1};
    auto v = std::vector<int>{1,1,2,3,4,4,5,5,5};
    auto ans = sol.topKFrequent(v,9);
    for(auto el: ans){
        std::cout << el << ' ';
    }
    std::cout <<  std::endl;
    return 0;
}

